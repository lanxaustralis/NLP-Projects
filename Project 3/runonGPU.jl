#jl Use `Literate.notebook(juliafile, ".", execute=false)` to convert to notebook.

# # Neural Machine Translation
#
# **Reference:** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." In Advances in neural information processing systems, pp. 3104-3112. 2014. ([Paper](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks), [Sample code](https://github.com/tensorflow/nmt))
#import Pkg
using Pkg
using Knet, Test, Base.Iterators, IterTools, Random # , LinearAlgebra, StatsBase
using AutoGrad: @gcheck  # to check gradients, use with Float64
Knet.atype() = KnetArray{Float32}  # determines what Knet.param() uses.
macro size(z, s) # for debugging
    esc(:(@assert (size($z) == $s) string(summary($z), !=, $s))) # for debugging
end # for debugging

Pkg.add("Statistics")
import Statistics
using Statistics

Pkg.add("CuArrays")
Pkg.build("CuArrays")

using CuArrays: CuArrays, usage_limit

CuArrays.usage_limit[] = 8_000_000_000
BATCH_SIZE = 64

Pkg.update()
pkgs = Pkg.installed()

for package in keys(pkgs)
    if pkgs[package] == nothing
        pkgs[package] = VersionNumber("0.0.1")
    end
    println("Package name: ", package, " Version: ", pkgs[package])
end

#array_type = KnetArray # For GPU instances
#array_type=Array # For CPU instances


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

word2ind(dict, x) = get(dict, x, 1)

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
    Linear(
        param(outputsize, inputsize),
        param0(outputsize),
    )
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

# ## Part 0. Load data
#
# We will use the Turkish-English pair from the [TED Talks Dataset](https://github.com/neulab/word-embeddings-for-nmt) for our experiments.

datadir = "datasets/tr_to_en"

if !isdir(datadir)
    download(
        "http://www.phontron.com/data/qi18naacl-dataset.tar.gz",
        "qi18naacl-dataset.tar.gz",
    )
    run(`tar xzf qi18naacl-dataset.tar.gz`)
end

if !isdefined(Main, :tr_vocab)
    tr_vocab = Vocab("$datadir/tr.train", mincount = 5)
    en_vocab = Vocab("$datadir/en.train", mincount = 5)
    tr_train = TextReader("$datadir/tr.train", tr_vocab)
    en_train = TextReader("$datadir/en.train", en_vocab)
    tr_dev = TextReader("$datadir/tr.dev", tr_vocab)
    en_dev = TextReader("$datadir/en.dev", en_vocab)
    tr_test = TextReader("$datadir/tr.test", tr_vocab)
    en_test = TextReader("$datadir/en.test", en_vocab)
    @info "Testing data"
    @test length(tr_vocab.i2w) == 38126
    @test length(first(tr_test)) == 16
    @test length(collect(tr_test)) == 5029
end


# ## Part 1. Minibatching
#
# For minibatching we are going to design a new iterator: `MTData`. This iterator is built
# on top of two TextReaders `src` and `tgt` that produce parallel sentences for source and
# target languages.

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
    # Your code here
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

#-

@info "Testing MTData"
dtrn = MTData(tr_train, en_train)
ddev = MTData(tr_dev, en_dev)
dtst = MTData(tr_test, en_test)

x, y = first(dtst)

# @test length(collect(dtst)) == 48
# @test size.((x, y)) == ((128, 10), (128, 24))
@test x[1, 1] == tr_vocab.eos
@test x[1, end] != tr_vocab.eos
@test y[1, 1] == en_vocab.eos
@test y[1, 2] != en_vocab.eos
@test y[1, end] == en_vocab.eos


# ## Part 2. Sequence to sequence model without attention
#
# In this part we will define a simple sequence to sequence encoder-decoder model for
# machine translation.

struct S2S_v1
    srcembed::Embed     # source language embedding
    encoder::RNN        # encoder RNN (can be bidirectional)
    tgtembed::Embed     # target language embedding
    decoder::RNN        # decoder RNN
    projection::Linear  # converts decoder output to vocab scores
    dropout::Real       # dropout probability to prevent overfitting
    srcvocab::Vocab     # source language vocabulary
    tgtvocab::Vocab     # target language vocabulary
end

function S2S_v1(
    hidden::Int,         # hidden size for both the encoder and decoder RNN
    srcembsz::Int,       # embedding size for source language
    tgtembsz::Int,       # embedding size for target language
    srcvocab::Vocab,     # vocabulary for source language
    tgtvocab::Vocab;     # vocabulary for target language
    layers = 1,            # number of layers
    bidirectional = false, # whether encoder RNN is bidirectional
    dropout = 0,
)           # dropout probability


    layerMultiplier = bidirectional ? 2 : 1

    S2S_v1(
        Embed(length(srcvocab.i2w), srcembsz),
        RNN(
            srcembsz,
            hidden,
            numLayers = layers,
            bidirectional = bidirectional,
            dropout = dropout
        ),
        Embed(length(tgtvocab.i2w), tgtembsz),
        RNN(
            tgtembsz,
            hidden,
            numLayers = layerMultiplier * layers,
            dropout = dropout
        ),
        Linear(hidden, length(tgtvocab.i2w)),
        dropout,
        srcvocab,
        tgtvocab,
    )

end


function (s::S2S_v1)(src, tgt; average = true)
    #B,Tx = size(src,2)
    B, Ty = size(tgt)
    Ty -= 1 # Crop one
    # Ex, Ey = length(model.srcembed([1])), length(model.tgtembed([1]))

    rnn_encoder = s.encoder
    rnn_decoder = s.decoder
    project = s.projection

    # Lx, Ly = rnn_encoder.numLayers, rnn_decoder.numLayers
    # Hx, Hy = rnn_encoder.hiddenSize, rnn_decoder.hiddenSize
    # Dx = Ly/Lx

    emb_out_src = s.srcembed(src)
    #@test size(emb_out_src)== (Ex,B,Tx) # Done

    # Safe for repetitive usage
    rnn_encoder.h = 0
    rnn_encoder.c = 0

    y_enc = rnn_encoder(emb_out_src)
    #@test size(y_enc) == (Hx*Dx,B,Tx)
    h_enc = rnn_encoder.h
    #@test size(h_enc) == (Hx,B,Lx*Dx)
    c_enc = rnn_encoder.c

    emb_out_tgt = s.tgtembed(tgt[:, 1:end-1])
    #@test size(emb_out_tgt)== (Ey,B,Ty)

    rnn_decoder.h = h_enc
    rnn_decoder.c = c_enc
    y_dec = rnn_decoder(emb_out_tgt)
    #@test size(y_dec)==(Hy,B,Ty)

    project_inp = reshape(y_dec, :, B * Ty)
    project_out = project(project_inp)

    #@test size(project_out)==(length(project.b),B*Ty)

    verify = deepcopy(tgt)
    mask!(verify, s.tgtvocab.eos)

    average && return mean(nll(project_out, verify[:, 2:end]))
    return nll(project_out, verify[:, 2:end]; average = false)
end

#-

@info "Testing S2S_v1"
Knet.seed!(1)
model = S2S_v1(
    512,
    512,
    512,
    tr_vocab,
    en_vocab;
    layers = 2,
    bidirectional = true,
    dropout = 0.2,
)
(x, y) = first(dtst)
## Your loss can be slightly different due to different ordering of words in the vocabulary.
## The reference vocabulary starts with eos, unk, followed by words in decreasing frequency.
#@test model(x,y; average=false) == (14097.471f0, 1432)  !!!!!!
println(model(x, y; average = false))


# ### Loss for a whole dataset
#
# Define a `loss(model, data)` which returns a `(Σloss, Nloss)` pair if `average=false` and
# a `Σloss/Nloss` average if `average=true` for a whole dataset. Assume that `data` is an
# iterator of `(x,y)` pairs such as `MTData` and `model(x,y;average)` is a model like
# `S2S_v1` that computes loss on a single `(x,y)` pair.

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

#-

@info "Testing loss"
@time res = loss(model, dtst, average = false)
println(res)
#@test res == (1.0429117f6, 105937) !!!!!!!!!!
## Your loss can be slightly different due to different ordering of words in the vocabulary.
## The reference vocabulary starts with eos, unk, followed by words in decreasing frequency.
## Also, because we do not mask src, different batch sizes may lead to slightly different
## losses. The test above gives (1.0429178f6, 105937) with batchsize==1.

# ### Training SGD_v1
#
# The following function can be used to train our model. `trn` is the training data, `dev`
# is used to determine the best model, `tst...` can be zero or more small test datasets for
# loss reporting. It returns the model that does best on `dev`.

function train!(model, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps = 100) do y
        losses = [loss(model, d) for d in (dev, tst...)]
        if losses[1] < bestloss
            bestmodel, bestloss = deepcopy(model), losses[1]
        end
        return (losses...,)
    end
    return bestmodel
end

# You should be able to get under 3.40 dev loss with the following settings in 10
# epochs. The training speed on a V100 is about 3 mins/epoch or 40K words/sec, K80 is about
# 6 times slower. Using settings closer to the Luong paper (per-sentence loss rather than
# per-word loss, SGD with lr=1, gclip=1 instead of Adam), you can get to 3.17 dev loss in
# about 25 epochs. Using dropout and shuffling batches before each epoch significantly
# improve the dev loss. You can play around with hyperparameters but I doubt results will
# get much better without attention. To verify your training, here is the dev loss I
# observed at the beginning of each epoch in one training session:
# `[9.83, 4.60, 3.98, 3.69, 3.52, 3.41, 3.35, 3.32, 3.30, 3.31, 3.33]`

@info "Training S2S_v1"
epochs = 10
ctrn = collect(dtrn)
trnx10 = collect(flatten(shuffle!(ctrn) for i = 1:epochs))
trn20 = ctrn[1:20]
dev38 = collect(ddev)
## Uncomment this to train the model (This takes about 30 mins on a V100):
## model = train!(model, trnx10, dev38, trn20)
## Uncomment this to save the model:
## Knet.save("s2s_v1.jld2","model",model)
## Uncomment this to load the model:
## model = Knet.load("s2s_v1.jld2","model")

# ### Generating translations
#
# With a single argument, a `S2S_v1` object should take it as a batch of source sentences
# and generate translations for them. After passing `src` through the encoder and copying
# its hidden states to the decoder, the decoder is run starting with an initial input of all
# `eos` tokens. Highest scoring tokens are appended to the output and used as input for the
# subsequent decoder steps.  The decoder should stop generating when all sequences in the
# batch have generated `eos` or when `stopfactor * size(src,2)` decoder steps are reached. A
# correctly shaped target language batch should be returned.

function (s::S2S_v1)(src::Matrix{Int}; stopfactor = 3)
    # Preperation for initial step
    B = size(src, 1)
    tgt = fill(s.tgtvocab.eos, (B, 1)) # size as (B,2)
    output = deepcopy(tgt)

    rnn_encoder = s.encoder
    rnn_decoder = s.decoder
    project = s.projection

    emb_out_src = s.srcembed(src)

    # Safe for repetitive usage
    rnn_encoder.h = 0
    rnn_encoder.c = 0

    y_enc = rnn_encoder(emb_out_src)
    h_enc = rnn_encoder.h
    c_enc = rnn_encoder.c

    rnn_decoder.h = h_enc
    rnn_decoder.c = c_enc

    step = 1
    max_step = stopfactor * size(src, 2)
    Ty = 1
    #@test Ty == size(tgt,2)

    while step <= max_step
        emb_out_tgt = s.tgtembed(tgt)

        y_dec = rnn_decoder(emb_out_tgt)

        project_inp = reshape(y_dec, :, B * Ty)
        project_out = project(project_inp)

        scores = softmax(project_out)

        for i = 1:B
            # Assigns the position of the highest token
            col = scores[:, i]
            colMax = col[1]
            index = 1
            for j in 1:length(col)
                if colMax<col[j]
                    colMax, index = col[j],j
                end
            end
            tgt[i] = index
        end

        all_eos = true
        for i in 1:length(tgt)
            if tgt[i]!=s.tgtvocab.eos
                all_eos = false
            end
        end

        all_eos && break # all produced eos

        output = hcat(output, tgt)
        step += 1
    end

    return output[:, 2:end]

end

#-

## Utility to convert int arrays to sentence strings
function int2str(y, vocab)
    y = vec(y)
    ysos = findnext(w -> !isequal(w, vocab.eos), y, 1)
    ysos == nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1 + length(y))
    join(vocab.i2w[y[ysos:yeos-1]], " ")
end

#-

@info "Generating some translations"
d = MTData(tr_dev, en_dev, batchsize = 1) |> collect
(src, tgt) = rand(d)
out = model(src)
println("SRC: ", int2str(src, model.srcvocab))
println("REF: ", int2str(tgt, model.tgtvocab))
println("OUT: ", int2str(out, model.tgtvocab))
## Here is a sample output:
## SRC: çin'e 15 şubat 2006'da ulaştım .
## REF: i made it to china on february 15 , 2006 .
## OUT: i got to china , china , at the last 15 years .

# ### Calculating BLEU
#
# BLEU is the most commonly used metric to measure translation quality. The following should
# take a model and some data, generate translations and calculate BLEU.

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

# Calculating dev BLEU takes about 45 secs on a V100. We get about 8.0 BLEU which is pretty
# low. As can be seen from the sample translations a loss of ~3+ (perplexity ~20+) or a BLEU
# of ~8 is not sufficient to generate meaningful translations.

@info "Calculating BLEU"
bleu(model, ddev)

# To improve the quality of translations we can use more training data, different training
# and model parameters, or preprocess the input/output: e.g. splitting Turkish words to make
# suffixes look more like English function words may help. Other architectures,
# e.g. attention and transformer, perform significantly better than this simple S2S model.
