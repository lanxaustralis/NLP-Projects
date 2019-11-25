#jl Use `Literate.notebook(juliafile, ".", execute=false)` to convert to notebook.

# # Attention-based Neural Machine Translation
#
# **Reference:** Luong, Thang, Hieu Pham and Christopher D. Manning. "Effective Approaches to Attention-based Neural Machine Translation." In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 1412-1421. 2015.
#
# * https://www.aclweb.org/anthology/D15-1166/ (main paper reference)
# * https://arxiv.org/abs/1508.04025 (alternative paper url)
# * https://github.com/tensorflow/nmt (main code reference)
# * https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention (alternative code reference)
# * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py:2449,2103 (attention implementation)

using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, IterTools#, CuArrays

import Pkg

# GPU related
# Pkg.add("CuArrays")
# Pkg.build("CuArrays")
#
# using CuArrays: CuArrays, usage_limit

Pkg.update()
pkgs = Pkg.installed()
Knet.atype() = Array{Float32}

for package in keys(pkgs)
    if pkgs[package] == nothing
        pkgs[package] = VersionNumber("0.0.1")
    end
    println("Package name: ", package, " Version: ", pkgs[package])
end

# Constants
BATCH_SIZE = 64

# ## Code and data from previous projects
#
# Please copy or include the following types and related functions from previous projects:
# `Vocab`, `TextReader`, `MTData`, `Embed`, `Linear`, `mask!`, `loss`, `int2str`,
# `bleu`.

## Your code here

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(
    file::String;
    tokenizer = split,
    vocabsize = Inf,
    mincount = 1,
    unk = "<unk>",
    eos = "<s>",
)
    vocab_freq = Dict{String,Int64}(unk => 1, eos => 1)
    w2i = Dict{String,Int64}(unk => 2, eos => 1)
    i2w = Vector{String}()

    push!(i2w, eos)
    push!(i2w, unk)

    open(file) do f
        for line in eachline(f)
            sentence = strip(lowercase(line))
            sentence = tokenizer(line, [' '], keepempty = false)

            for word in sentence
                word == unk && continue
                word == eos && continue # They are default ones to be added later
                vocab_freq[word] = get!(vocab_freq, word, 0) + 1
            end
        end
        close(f)
    end


# End of vanilla implementation of the vocaulary
# From here we must add the mincount and vocabsize properties
# We must change the first two property of the vocab wrt those paramaters
    vocab_freq = sort!(
        collect(vocab_freq),
        by = tuple -> last(tuple),
        rev = true,
    )

    if length(vocab_freq) > vocabsize - 2 # eos and unk ones
        vocab_freq = vocab_freq[1:vocabsize-2] # trim to fit the size
    end

#vocab_freq = reverse(vocab_freq)

    while true
        length(vocab_freq) == 0 && break
        word, freq = vocab_freq[end]
        freq >= mincount && break # since it is already ordered
        vocab_freq = vocab_freq[1:(end-1)]
    end
#pushfirst!(vocab_freq,unk=>1,eos=>1) # freq does not matter, just adding the
    for i = 1:length(vocab_freq)
        word, freq = vocab_freq[i]
        ind = (get!(w2i, word, 1 + length(w2i)))
        (length(i2w) < ind) && push!(i2w, word)
    end

    return Vocab(w2i, i2w, 2, 1, tokenizer)
end

struct TextReader
    file::String
    vocab::Vocab
end

word2ind(dict, x) = get(dict, x, 2) # unk is 2

#Implementing the iterate function
function Base.iterate(r::TextReader, s = nothing)
    if s == nothing
        state = open(r.file)
        Base.iterate(r, state)
    else
        if eof(s) == true
            close(s)
            return nothing
        else
            line = readline(s)
            line = strip(lowercase(line))
            sent = r.vocab.tokenizer(line, [' '], keepempty = false)
            sent_ind = Int[]
            for word in sent
                ind = word2ind(r.vocab.w2i, word)
                push!(sent_ind, ind)
            end
            return (sent_ind, s)
        end
    end
end


Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

#Embed
struct Embed
    w
end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize, vocabsize))
end

function (l::Embed)(x)
    l.w[:, x]
end

#Linear
struct Linear
    w
    b
end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize), param0(outputsize))
end

function (l::Linear)(x)
    l.w * mat(x, dims = 1) .+ l.b
end

# Mask!
function mask!(a, pad)
    matr = a
    for j = 1:size(matr)[1]
        i = 0
        while (i < length(matr[j, :]) - 1)
            if matr[j, length(matr[j, :])-i-1] != pad
                break

            elseif matr[j, length(matr[j, :])-i] == pad
                matr[j, length(matr[j, :])-i] = 0
            end
            i += 1
        end
    end
    return matr
end

struct MTData
    src::TextReader        # reader for source language data
    tgt::TextReader        # reader for target language data
    batchsize::Int         # desired batch size
    maxlength::Int         # skip if source sentence above maxlength
    batchmajor::Bool       # batch dims (B,T) if batchmajor=false (default) or (T,B) if true.
    bucketwidth::Int       # batch sentences with length within bucketwidth of each other
    buckets::Vector        # sentences collected in separate arrays called buckets for each length range
    batchmaker::Function   # function that turns a bucket into a batch.
end

function MTData(
    src::TextReader,
    tgt::TextReader;
    batchmaker = arraybatch,
    batchsize = BATCH_SIZE,
    maxlength = typemax(Int),
    batchmajor = false,
    bucketwidth = 10,
    numbuckets = min(BATCH_SIZE, maxlength ÷ bucketwidth),
)
    buckets = [[] for i = 1:numbuckets] # buckets[i] is an array of sentence pairs with similar length
    MTData(
        src,
        tgt,
        batchsize,
        maxlength,
        batchmajor,
        bucketwidth,
        buckets,
        batchmaker,
    )
end

Base.IteratorSize(::Type{MTData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MTData}) = Base.HasEltype()
Base.eltype(::Type{MTData}) = NTuple{2}

function Base.iterate(d::MTData, state = nothing)
    if state == nothing
        for b in d.buckets
            empty!(b)
        end
        state_src, state_tgt = nothing, nothing
    else
        state_src, state_tgt = state
    end
    bucket, ibucket = nothing, nothing


    while true
        iter_src = (state_src === nothing ? iterate(d.src) :
                    iterate(d.src, state_src))
        iter_tgt = (state_tgt === nothing ? iterate(d.tgt) :
                    iterate(d.tgt, state_tgt))

        if iter_src === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent_src, state_src = iter_src
            sent_tgt, state_tgt = iter_tgt
            if length(sent_src) > d.maxlength || length(sent_src) == 0
                continue
            end
            ibucket = min(
                1 + (length(sent_src) - 1) ÷ d.bucketwidth,
                length(d.buckets),
            )
            bucket = d.buckets[ibucket]
            push!(bucket, (sent_src, sent_tgt))
            if length(bucket) === d.batchsize
                break
            end
        end
    end
    if bucket === nothing
        return nothing
    end

    batch = d.batchmaker(d, bucket)

    empty!(bucket)
    return batch, (state_src, state_tgt)
end


function arraybatch(d::MTData, bucket)
    bucketx = map(x -> x[1], bucket)
    buckety = map(x -> x[2], bucket)
    batch_x = fill(d.src.vocab.eos, length(bucketx), maximum(length.(bucketx)))
    for i = 1:length(bucket)
        batch_x[i, end-length(bucketx[i])+1:end] = bucketx[i]
    end
    batch_y = fill(
        d.tgt.vocab.eos,
        length(buckety),
        maximum(length.(buckety)) + 2,
    )
    for i = 1:length(bucket)
        batch_y[i, 2:length(buckety[i])+1] = buckety[i]
    end

    return (batch_x, batch_y)
end

function loss(model, data; average = true)
    total_loss = 0
    total_word = 0

    for (x, y) in collect(data)
        curr_loss, curr_word = model(x, y; average = false)
        total_loss += curr_loss
        total_word += curr_word
    end

    average && return total_loss / total_word
    return (total_loss, total_word)

end
function int2str(y, vocab)
    y = vec(y)
    ysos = findnext(w -> !isequal(w, vocab.eos), y, 1)
    ysos == nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1 + length(y))
    join(vocab.i2w[y[ysos:yeos-1]], " ")
end
function bleu(s2s, d::MTData)
    d = MTData(d.src, d.tgt, batchsize = 1)
    reffile = d.tgt.file
    hypfile, hyp = mktemp()
    for (x, y) in progress(collect(d))
        g = s2s(x)
        for i = 1:size(y, 1)
            println(hyp, int2str(g[i, :], d.tgt.vocab))
        end
    end
    close(hyp)
    isfile("multi-bleu.perl") || download(
        "https://github.com/moses-smt/mosesdecoder/raw/master/scripts/generic/multi-bleu.perl",
        "multi-bleu.perl",
    )
    run(pipeline(`cat $hypfile`, `perl multi-bleu.perl $reffile`))
    return hypfile
end

## Pre Assigment Part Completed


# ## S2S: Sequence to sequence model with attention
#
# In this project we will define, train and evaluate a sequence to sequence encoder-decoder
# model with attention for Turkish-English machine translation. The model has two extra
# fields compared to `S2S_v1`: the `memory` layer computes keys and values from the encoder,
# the `attention` layer computes the attention vector for the decoder.

struct Memory
    w
end

struct Attention
    wquery
    wattn
    scale
end

struct S2S
    srcembed::Embed       # encinput(B,Tx) -> srcembed(Ex,B,Tx)
    encoder::RNN          # srcembed(Ex,B,Tx) -> enccell(Dx*H,B,Tx)
    memory::Memory        # enccell(Dx*H,B,Tx) -> keys(H,Tx,B), vals(Dx*H,Tx,B)
    tgtembed::Embed       # decinput(B,Ty) -> tgtembed(Ey,B,Ty)
    decoder::RNN          # tgtembed(Ey,B,Ty) . attnvec(H,B,Ty)[t-1] = (Ey+H,B,Ty) -> deccell(H,B,Ty)
    attention::Attention  # deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
    projection::Linear    # attnvec(H,B,Ty) -> proj(Vy,B,Ty)
    dropout::Real         # dropout probability
    srcvocab::Vocab       # source language vocabulary
    tgtvocab::Vocab       # target language vocabulary
end


# ## Load pretrained model and data
#
# We will load a pretrained model (16.20 bleu) for code testing.  The data should be loaded
# with the vocabulary from the pretrained model for word id consistency.

if !isdefined(Main, :pretrained) || pretrained === nothing
    @info "Loading reference model"
    isfile("s2smodel.jld2") || download(
        "http://people.csail.mit.edu/deniz/comp542/s2smodel.jld2",
        "s2smodel.jld2",
    )
    pretrained = Knet.load("s2smodel.jld2", "model")
end
datadir = "datasets/tr_to_en"
if !isdir(datadir)
    @info "Downloading data"
    download(
        "http://www.phontron.com/data/qi18naacl-dataset.tar.gz",
        "qi18naacl-dataset.tar.gz",
    )
    run(`tar xzf qi18naacl-dataset.tar.gz`)
end

if !isdefined(Main, :tr_vocab)
    BATCHSIZE, MAXLENGTH = 64, 50
    @info "Reading data"
    tr_vocab = pretrained.srcvocab # Vocab("$datadir/tr.train", mincount=5)
    en_vocab = pretrained.tgtvocab # Vocab("$datadir/en.train", mincount=5)
    tr_train = TextReader("$datadir/tr.train", tr_vocab)
    en_train = TextReader("$datadir/en.train", en_vocab)
    tr_dev = TextReader("$datadir/tr.dev", tr_vocab)
    en_dev = TextReader("$datadir/en.dev", en_vocab)
    tr_test = TextReader("$datadir/tr.test", tr_vocab)
    en_test = TextReader("$datadir/en.test", en_vocab)
    dtrn = MTData(
        tr_train,
        en_train,
        batchsize = BATCHSIZE,
        maxlength = MAXLENGTH,
    )
    ddev = MTData(tr_dev, en_dev, batchsize = BATCHSIZE)
    dtst = MTData(tr_test, en_test, batchsize = BATCHSIZE)
end

# ## Part 1. Model constructor
#
# The `S2S` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors for both the encoder and the decoder
# * `srcembsz`, `tgtembsz`: size of the source/target language embedding vectors
# * `srcvocab`, `tgtvocab`: the source/target language vocabulary
# * `layers=1`: number of layers
# * `bidirectional=false`: whether the encoder is bidirectional
# * `dropout=0`: dropout probability
#
# Hints:
# * You can find the vocabulary size with `length(vocab.i2w)`.
# * If the encoder is bidirectional `layers` must be even and the encoder should have `layers÷2` layers.
# * The decoder will use "input feeding", i.e. it will concatenate its previous output to its input. Therefore the input size for the decoder should be `tgtembsz+hidden`.
# * Only `numLayers`, `dropout`, and `bidirectional` keyword arguments should be used for RNNs, leave everything else default.
# * The memory parameter `w` is used to convert encoder states to keys. If the encoder is bidirectional initialize it to a `(hidden,2*hidden)` parameter, otherwise set it to the constant 1.
# * The attention parameter `wquery` is used to transform the query, set it to the constant 1 for this project.
# * The attention parameter `scale` is used to scale the attention scores before softmax, set it to a parameter of size 1.
# * The attention parameter `wattn` is used to transform the concatenation of the decoder output and the context vector to the attention vector. It should be a parameter of size `(hidden,2*hidden)` if unidirectional, `(hidden,3*hidden)` if bidirectional.

function S2S(hidden::Int, srcembsz::Int, tgtembsz::Int, srcvocab::Vocab, tgtvocab::Vocab;
             layers=1, bidirectional=false, dropout=0)
    ## Your code here
    layerMultiplier = bidirectional ? 2 : 1

    if bidirectional
        wattnsize = (hidden, 3 * hidden)
        mem = Memory(param(hidden, 2 * hidden))
    else
        wattnsize = (hidden, 2 * hidden)
        mem = Memory(1)
    end

    S2S(
        Embed(length(srcvocab.i2w), srcembsz),
        RNN(
            srcembsz,
            hidden,
            numLayers = (1 / layerMultiplier) * layers,
            bidirectional = bidirectional,
            dropout = dropout,
        ),
        mem,
        Embed(length(tgtvocab.i2w), tgtembsz),
        RNN(tgtembsz + hidden, hidden, numLayers = layers, dropout = dropout),
        Attention(1, param(wattnsize[1], wattnsize[2]), param(1)),
        Linear(hidden, length(tgtvocab.i2w)),
        dropout,
        srcvocab,
        tgtvocab,
    )

end

#-
@testset "Testing S2S constructor" begin
    H, Ex, Ey, Vx, Vy, L, Dx, Pdrop = 8,
        9,
        10,
        length(dtrn.src.vocab.i2w),
        length(dtrn.tgt.vocab.i2w),
        2,
        2,
        0.2
    m = S2S(
        H,
        Ex,
        Ey,
        dtrn.src.vocab,
        dtrn.tgt.vocab;
        layers = L,
        bidirectional = (Dx == 2),
        dropout = Pdrop,
    )
    @test size(m.srcembed.w) == (Ex, Vx)
    @test size(m.tgtembed.w) == (Ey, Vy)
    @test m.encoder.inputSize == Ex
    @test m.decoder.inputSize == Ey + H
    @test m.encoder.hiddenSize == m.decoder.hiddenSize == H
    @test m.encoder.direction == Dx - 1
    @test m.encoder.numLayers == (Dx == 2 ? L ÷ 2 : L)
    @test m.decoder.numLayers == L
    @test m.encoder.dropout == m.decoder.dropout == Pdrop
    @test size(m.projection.w) == (Vy, H)
    @test size(m.memory.w) == (Dx == 2 ? (H, 2H) : ())
    @test m.attention.wquery == 1
    @test size(m.attention.wattn) == (Dx == 2 ? (H, 3H) : (H, 2H))
    @test size(m.attention.scale) == (1,)
    @test m.srcvocab === dtrn.src.vocab
    @test m.tgtvocab === dtrn.tgt.vocab
end


# ## Part 2. Memory
#
# The memory layer turns the output of the encoder to a pair of tensors that will be used as
# keys and values for the attention mechanism. Remember that the encoder RNN output has size
# `(H*D,B,Tx)` where `H` is the hidden size, `D` is 1 for unidirectional, 2 for
# bidirectional, `B` is the batchsize, and `Tx` is the sequence length. It will be
# convenient to store these values in batch major form for the attention mechanism, so
# *values* in memory will be a permuted copy of the encoder output with size `(H*D,Tx,B)`
# (see `@doc permutedims`). The *keys* in the memory need to have the same first dimension
# as the *queries* (i.e. the decoder hidden states). So *values* will be transformed into
# *keys* of size `(H,B,Tx)` with `keys = m.w * values` where `m::Memory` is the memory
# layer. Note that you will have to do some reshaping to 2-D and back to 3-D for matrix
# multiplications. Also note that `m.w` may be a scalar such as `1` e.g. when `D=1` and we
# want keys and values to be identical.


function (m::Memory)(x)
    ## Your code here
    H, B, Tx = size(x)
    val = copy(x)
    v = permutedims(val, (1, 3, 2))
    k = mmul(m.w, v)
    return k, v
end

# You can use the following helper function for scaling and linear transformations of 3-D tensors:
mmul(w, x) = (w == 1 ? x :
 w == 0 ? 0 :
 reshape(w * reshape(x, size(x, 1), :), (:, size(x)[2:end]...)))

#-
@testset "Testing memory" begin
    H, D, B, Tx = pretrained.encoder.hiddenSize,
        pretrained.encoder.direction + 1,
        4,
        5
    #x = KnetArray(randn(Float32, H * D, B, Tx)) !! GPU
    x = Array(randn(Float32, H * D, B, Tx))
    k, v = pretrained.memory(x)
    @test v == permutedims(x, (1, 3, 2))
    @test k == mmul(pretrained.memory.w, v)
end


# ## Part 3. Encoder
#
# `encode()` takes a model `s` and a source language minibatch `src`. It passes the input
# through `s.srcembed` and `s.encoder` layers with the `s.encoder` RNN hidden states
# initialized to `0` in the beginning, and copied to the `s.decoder` RNN at the end. The
# steps so far are identical to `S2S_v1` but there is an extra step: The encoder output is
# passed to the `s.memory` layer which returns a `(keys,values)` pair. `encode()` returns
# this pair to be used later by the attention mechanism.

function encode(s::S2S, src)
    ## Your code here
    rnn_encoder = s.encoder
    rnn_decoder = s.decoder

    emb_out_src = s.srcembed(src)

   # Safe for repetitive usage
    rnn_encoder.h = 0
    rnn_encoder.c = 0

    y_enc = rnn_encoder(emb_out_src)
    rnn_decoder.h = rnn_encoder.h
    rnn_decoder.c = rnn_encoder.c
    return s.memory(y_enc)
end

#-
@testset "Testing encoder" begin
    src1, tgt1 = first(dtrn)
    key1, val1 = encode(pretrained, src1)
    H, D, B, Tx = pretrained.encoder.hiddenSize,
        pretrained.encoder.direction + 1,
        size(src1, 1),
        size(src1, 2)
    @test size(key1) == (H, Tx, B)
    @test size(val1) == (H * D, Tx, B)
    @test (pretrained.decoder.h, pretrained.decoder.c) === (
        pretrained.encoder.h,
        pretrained.encoder.c,
    )
    @test norm(key1) ≈ 1214.4755f0
    @test norm(val1) ≈ 191.10411f0
    @test norm(pretrained.decoder.h) ≈ 48.536964f0
    @test norm(pretrained.decoder.c) ≈ 391.69028f0
end


# ## Part 4. Attention
#
# The attention layer takes `cell`: the decoder output, and `mem`: a pair of (keys,vals)
# from the encoder, and computes and returns the attention vector. First `a.wquery` is used
# to linearly transform the cell to the query tensor. The query tensor is reshaped and/or
# permuted as appropriate and multiplied with the keys tensor to compute the attention
# scores. Please see `@doc bmm` for the batched matrix multiply operation used for this
# step. The attention scores are scaled using `a.scale` and normalized along the time
# dimension using `softmax`. After the appropriate reshape and/or permutation, the scores
# are multiplied with the `vals` tensor (using `bmm` again) to compute the context
# tensor. After the appropriate reshape and/or permutation the context vector is
# concatenated with the cell and linearly transformed to the attention vector using
# `a.wattn`. Please see the paper and code examples for details.
#
# Note: the paper mentions a final `tanh` transform, however the final version of the
# reference code does not use `tanh` and gets better results. Therefore we will skip `tanh`.

#deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
function (a::Attention)(cell, mem)
    ## Your code here
    H,B,Ty = size(cell)

    qtensor = cell*a.wquery

    qtensor = permutedims(qtensor, (3,1,2))

    scores = bmm(qtensor,mem[1]) # Multiply with keys

    scores=(a.scale[1])*scores
    scores = softmax(scores,dims=2)

    v= permutedims(mem[2],(2,1,3))
    context = bmm(scores,v)


    context = permutedims(context,(2,3,1))
    context = cat(cell,context,dims = 1)

    context = reshape(context,(size(a.wattn)[2],:))
    context = a.wattn*context
    context = reshape(context,(H,B,Ty))
end

#-
@testset "Testing attention" begin
    src1, tgt1 = first(dtrn)
    key1, val1 = encode(pretrained, src1)
    H, B = pretrained.encoder.hiddenSize, size(src1, 1)
    Knet.seed!(1)
    #x = KnetArray(randn(Float32, H, B, 5)) !! GPU
    x = Array(randn(Float32, H, B, 5))
    y = pretrained.attention(x, (key1, val1))
    @test size(y) == size(x)
    @test norm(y) ≈ 808.381f0
end


# ## Part 5. Decoder
#
# `decode()` takes a model `s`, a target language minibatch `tgt`, the memory from the
# encoder `mem` and the decoder output from the previous time step `prev`. After the input
# is passed through the embedding layer, it is concatenated with `prev` (this is called
# input feeding). The resulting tensor is passed through `s.decoder`. Finally the
# `s.attention` layer takes the decoder output and the encoder memory to compute the
# "attention vector" which is returned by `decode()`.

function decode(s::S2S, tgt, mem, prev)
    # Your code here
    rnn_decoder = s.decoder
    rnn_encoder =s.encoder
    emb_out_tgt = s.tgtembed(tgt)

    inputfeeding =cat(emb_out_tgt,prev,dims =1)

    rnn_decoder.h = rnn_encoder.h
    rnn_decoder.c = rnn_encoder.c
    y_dec = rnn_decoder(inputfeeding)

    attentionout = s.attention(y_dec,mem)
end

#-
@testset "Testing decoder" begin
    src1, tgt1 = first(dtrn)
    key1, val1 = encode(pretrained, src1)
    H, B = pretrained.encoder.hiddenSize, size(src1, 1)
    Knet.seed!(1)
    cell = randn!(similar(key1, size(key1, 1), size(key1, 3), 1))
    cell = decode(pretrained, tgt1[:, 1:1], (key1, val1), cell)
    @test size(cell) == (H, B, 1)
    @test norm(cell) ≈ 131.21631f0
end


# ## Part 6. Loss
#
# The loss function takes source language minibatch `src`, and a target language minibatch
# `tgt` and returns `sumloss/numwords` if `average=true` or `(sumloss,numwords)` if
# `average=false` where `sumloss` is the total negative log likelihood loss and `numwords` is
# the number of words predicted (including a final eos for each sentence). The source is first
# encoded using `encode` yielding a `(keys,vals)` pair (memory). Then the decoder is called to
# predict each word of `tgt` given the previous word, `(keys,vals)` pair, and the previous
# decoder output. The previous decoder output is initialized with zeros for the first
# step. The output of the decoder at each step is passed through the projection layer giving
# word scores. Losses can be computed from word scores and masked/shifted `tgt`.

function (s::S2S)(src, tgt; average = true)
    ## Your code here
    B,Ty = size(tgt)
    Ty -=1

    project = s.projection

    mem = encode(s,src)
    prev = zeros(Float32,s.decoder.hiddenSize,B,1)

    verify = deepcopy(tgt[:,2:end])
    mask!(verify, s.tgtvocab.eos)

    total_loss = 0
    total_word = 0

    for word_order in 1:Ty
        dec_state = decode(s, tgt[:,word_order], mem, prev)
        pred = project(reshape(dec_state, :, B))
        Δloss,Δword = nll(pred,verify[:,word_order];average=false)
        total_loss += Δloss
        total_word += Δword
        prev = dec_state
    end

    # prev = decode(s, tgt[:,1:(end-1)], mem, prev)
    # output = project(reshape(prev, :, B*Ty))



    average && return total_loss*1.0/total_word
    return total_loss,total_word
end

#-
@testset "Testing loss" begin
    src1, tgt1 = first(dtrn)
    @test pretrained(src1, tgt1) ≈ 1.4666592f0
    @test pretrained(src1, tgt1, average = false) == (1949.1901f0, 1329)
end

# ## Part 7. Greedy translator
#
# An `S2S` object can be called with a single argument (source language minibatch `src`, with
# size `B,Tx`) to generate translations (target language minibatch with size `B,Ty`). The
# keyword argument `stopfactor` determines how much longer the output can be compared to the
# input. Similar to the loss function, the source minibatch is encoded yield a `(keys,vals)`
# pair (memory). We generate the output one time step at a time by calling the decoder with
# the last output, the memory, and the last decoder state. The last output is initialized to
# an array of `eos` tokens and the last decoder state is initialized to an array of
# zeros. After computing the scores for the next word using the projection layer, the highest
# scoring words are selected and appended to the output. The generation stops when all outputs
# in the batch have generated `eos` or when the length of the output is `stopfactor` times the
# input.

function (s::S2S)(src; stopfactor = 3)
    # Your code here
end

#-
@testset "Testing translator" begin
    src1, tgt1 = first(dtrn)
    tgt2 = pretrained(src1)
    @test size(tgt2) == (64, 41)
    @test tgt2[1:3, 1:3] == [14 25 10647; 37 25 1426; 27 5 349]
end


# ## Part 8. Training
#
# `trainmodel` creates, trains and returns an `S2S` model. The arguments are described in
# comments.

function trainmodel(
    trn,                  # Training data
    dev,                  # Validation data, used to determine the best model
    tst...;               # Zero or more test datasets, their loss will be periodically reported
    bidirectional = true, # Whether to use a bidirectional encoder
    layers = 2,           # Number of layers (use `layers÷2` for a bidirectional encoder)
    hidden = 512,         # Size of the hidden vectors
    srcembed = 512,       # Size of the source language embedding vectors
    tgtembed = 512,       # Size of the target language embedding vectors
    dropout = 0.2,        # Dropout probability
    epochs = 0,           # Number of epochs (one of epochs or iters should be nonzero for training)
    iters = 0,            # Number of iterations (one of epochs or iters should be nonzero for training)
    bleu = false,         # Whether to calculate the BLEU score for the final model
    save = false,         # Whether to save the final model
    seconds = 60,         # Frequency of progress reporting
)
    @show bidirectional,
        layers,
        hidden,
        srcembed,
        tgtembed,
        dropout,
        epochs,
        iters,
        bleu,
        save
    flush(stdout)
    model = S2S(
        hidden,
        srcembed,
        tgtembed,
        trn.src.vocab,
        trn.tgt.vocab;
        layers = layers,
        dropout = dropout,
        bidirectional = bidirectional,
    )

    epochs == iters == 0 && return model

    (ctrn, cdev, ctst) = collect(trn), collect(dev), collect.(tst)
    traindata = (epochs > 0 ?
                 collect(flatten(shuffle!(ctrn) for i = 1:epochs)) :
                 shuffle!(collect(take(cycle(ctrn), iters))))

    bestloss, bestmodel = loss(model, cdev), deepcopy(model)
    progress!(adam(model, traindata), seconds = seconds) do y
        devloss = loss(model, cdev)
        tstloss = map(d -> loss(model, d), ctst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (dev = devloss, tst = tstloss, mem = Float32(CuArrays.usage[]))
    end
    save && Knet.save("attn-$(Int(time_ns())).jld2", "model", bestmodel)
    bleu && Main.bleu(bestmodel, dev)
    return bestmodel
end

# Train a model: If your implementation is correct, the first epoch should take about 24
# minutes on a v100 and bring the loss from 9.83 to under 4.0. 10 epochs would take about 4
# hours on a v100. With other GPUs you may have to use a smaller batch size (if memory is
# lower) and longer time (if gpu speed is lower).

## Uncomment the appropriate option for training:
model = pretrained  # Use reference model
## model = Knet.load("attn-1538395466294882.jld2", "model")  # Load pretrained model
## model = trainmodel(dtrn,ddev,take(dtrn,20); epochs=10, save=true, bleu=true)  # Train model

# Code to sample translations from a dataset
data1 = MTData(tr_dev, en_dev, batchsize = 1) |> collect;
function translate_sample(model, data)
    (src, tgt) = rand(data)
    out = model(src)
    println("SRC: ", int2str(src, model.srcvocab))
    println("REF: ", int2str(tgt, model.tgtvocab))
    println("OUT: ", int2str(out, model.tgtvocab))
end

# Generate translations for random instances from the dev set
translate_sample(model, data1)

# Code to generate translations from user input
function translate_input(model)
    v = model.srcvocab
    src = [get(v.w2i, w, v.unk) for w in v.tokenizer(readline())]'
    out = model(src)
    println("SRC: ", int2str(src, model.srcvocab))
    println("OUT: ", int2str(out, model.tgtvocab))
end

# Generate translations for user input
## translate_input(model)

# ## Competition
#
# The reference model `pretrained` has 16.2 bleu. By playing with the optimization algorithm
# and hyperparameters, using per-sentence loss, and (most importantly) splitting the Turkish
# words I was able to push the performance to 21.0 bleu. I will give extra credit to groups
# that can exceed 21.0 bleu in this dataset.
