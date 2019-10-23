import Pkg
using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging
Pkg.update()
pkgs = Pkg.installed()

for package in keys(pkgs)
    if pkgs[package] == nothing
        pkgs[package] = VersionNumber("0.0.1")
    end
    println("Package name: ", package, " Version: ", pkgs[package])
end

const datadir = "nn4nlp-code/data/ptb"
isdir(datadir) || run(`git clone https://github.com/neubig/nn4nlp-code.git`)

# Change wrt GPU instances
# param(dims...) = Param(KnetArray(0.01f0 * randn(Float32, dims...)))
# Knet.param(dims...) = Param(Array(0.01f0 * randn(Float32, dims...)))
# Values are not the same as they are supposed to be, hence we use the inner atype

array_type=KnetArray # For GPU instances
#array_type=Array # For CPU instances

# The Abstraction of the Vocabulary
struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

# Implementing the Vocabulary
function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    vocab_freq = Dict{String,Int64}(unk => 1, eos => 1)
    w2i = Dict{String, Int64}(unk => 1, eos => 2)
    i2w = Vector{String}()
    push!(i2w, unk)
    push!(i2w, eos)

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

    if length(vocab_freq)>vocabsize - 2 # eos and unk ones
        vocab_freq = vocab_freq[1:vocabsize-2] # trim to fit the size
    end

    vocab_freq = reverse(vocab_freq)

    while true
        length(vocab_freq)==0 && break
        word,freq = vocab_freq[1]
        freq>=mincount && break # since it is already ordered
        vocab_freq = vocab_freq[2:length(vocab_freq)]
    end
    #pushfirst!(vocab_freq,unk=>1,eos=>1) # freq does not matter, just adding the
    for i in 1:length(vocab_freq)
        word, freq = vocab_freq[i]
        ind = (get!(w2i, word, 1+length(w2i)))
        (length(i2w) < ind) && push!(i2w, word)
    end

    return Vocab(w2i, i2w, 1, 2, tokenizer)
end

# Testing the Vocabulary
@info "Testing Vocab"
f = "$datadir/train.txt"
v = Vocab(f)
@test all(v.w2i[w] == i for (i,w) in enumerate(v.i2w))
@test length(Vocab(f).i2w) == 10000
@test length(Vocab(f, vocabsize=1234).i2w) == 1234
@test length(Vocab(f, mincount=5).i2w) == 9859

train_vocab = v # The backbone vocab of the model

# Abstraction of the TextReader
struct TextReader
    file::String
    vocab::Vocab
end

word2ind(dict,x) = get(dict, x, 1)

#Implementing the iterate function
function Base.iterate(r::TextReader, s=nothing)
    if s == nothing
        state = open(r.file)
        Base.iterate(r,state)
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
                ind = word2ind(r.vocab.w2i,word)
                push!(sent_ind,ind)
            end
            return (sent_ind, s)
        end
    end
end

# Optional Functions for Iterator
Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

# Testing Iterator
@info "Testing TextReader"
train_sentences, valid_sentences, test_sentences =
    (TextReader("$datadir/$file.txt", train_vocab) for file in ("train","valid","test"))
@test length(first(train_sentences)) == 24
@test length(collect(train_sentences)) == 42068
@test length(collect(valid_sentences)) == 3370
@test length(collect(test_sentences)) == 3761

# Abstraction of the Embeding
struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    return Embed(param(embedsize,vocabsize;atype=array_type))
end

function (l::Embed)(x)
    # W = l.w
    # L = param((size(W))[1],size(x)...;atype=array_type)
    # if ndims(x)==1
    #     L = W[:,x]
    # else
    #     # for col in 1:size(x)[2]
    #     #     L[:,:,col] = W[:,x[:,col]]
    #     # end
    #     L = W[:,x[:,:]]
    # end

    return l.w[:,x] # it seems far efficient :)
end

# Testing the Embed
@info "Testing Embed"
Knet.seed!(1)
embed = Embed(100,10)
input = rand(1:100, 2, 3)
output = embed(input)
@test size(output) == (10, 2, 3)
@test norm(output) ≈ 0.59804f0

# Abstraction of the Linear
struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    w = param(outputsize,inputsize;atype=array_type)
    b = param0(outputsize;atype=array_type)
    return Linear(w,b)
end

function (l::Linear)(x)
    l.w * x .+ l.b
end

# Testing the Linear
@info "Testing Linear"
Knet.seed!(1)
linear = Linear(100,10)
input = oftype(linear.w, randn(Float32, 100, 5))
output = linear(input)
@test size(output) == (10, 5)
@test norm(output) ≈ 5.5301356f0

#Abstraction of the NNLM
struct NNLM; vocab; windowsize; embed; hidden; output; dropout; end


function NNLM(vocab::Vocab, windowsize::Int, embedsize::Int, hiddensize::Int, dropout::Real)
    vocabsize = length(v.i2w)
    ## NNLM (vocab::Vocab,  windowsize::Int,  embed::Embed,  hidden::Linear,  output::Linear,  dropout::Int)
    NNLM(vocab, windowsize, Embed(vocabsize,embedsize),Linear(embedsize*windowsize,hiddensize),Linear(hiddensize,vocabsize),dropout)
end

#-

## Default model parameters
HIST = 3
EMBED = 128
HIDDEN = 128
DROPOUT = 0.5
VOCAB = length(train_vocab.i2w)

# Testing the NNLM
@info "Testing NNLM"
model = NNLM(train_vocab, HIST, EMBED, HIDDEN, DROPOUT)
@test model.vocab === train_vocab
@test model.windowsize === HIST
@test size(model.embed.w) == (EMBED,VOCAB)
@test size(model.hidden.w) == (HIDDEN,HIST*EMBED)
@test size(model.hidden.b) == (HIDDEN,)
@test size(model.output.w) == (VOCAB,HIDDEN)
@test size(model.output.b) == (VOCAB,)
@test model.dropout == 0.5

## Pred v1
function pred_v1(m::NNLM, hist::AbstractVector{Int})
    @assert length(hist) == m.windowsize

    emb_inp = hist
    emb_out = m.embed(emb_inp)
    emb_out = dropout(emb_out,m.dropout;seed=1)

    hid_inp = vec(emb_out)
    hid_out = tanh.(m.hidden(hid_inp))
    hid_out = dropout(hid_out,m.dropout;seed=1)


    out = m.output(hid_out)

    return out
end

# Testing predv1
@info "Testing pred_v1"
h = repeat([model.vocab.eos], model.windowsize)
p = pred_v1(model, h)
@test size(p) == size(train_vocab.i2w)

## This predicts the scores for the whole sentence, will be used for later testing.
function scores_v1(model, sent)
    hist = repeat([ model.vocab.eos ], model.windowsize)
    scores = []
    for word in [ sent; model.vocab.eos ]
        push!(scores, pred_v1(model, hist))
        hist = [ hist[2:end]; word ]
    end
    hcat(scores...)
end

# Testing with scores
sent = first(train_sentences)
@test size(scores_v1(model, sent)) == (length(train_vocab.i2w), length(sent)+1)

# Function to generate new words
function generate(m::NNLM; maxlength=30)
    hist = repeat([m.vocab.eos],m.windowsize)
    vocab = m.vocab
    word_ids = []
    for i in 1:maxlength
        scores = softmax(pred_v1(m,hist))
        new_word_index = vocab.w2i[sample(vocab.i2w,weights(scores))]

        if new_word_index == m.vocab.eos
            break
        end
        push!(word_ids,new_word_index)
        hist = [ hist[2:end]; new_word_index ]
    end
    return join(vocab.i2w[word_ids]," ")

end

# Testing generate function
# @info "Testing generate"
# s = generate(model, maxlength=5)
# @test s isa String
# @test length(split(s)) <= 5

## Loss function for predv1
using Statistics
function loss_v1(m::NNLM, sent::AbstractVector{Int}; average = true)
    hist = repeat([ model.vocab.eos ], model.windowsize)
    total_loss = []
    count = 0
    for next_word in [ sent; model.vocab.eos ]
        s = pred_v1(model, hist)
        push!(total_loss, nll(s, [next_word]))
        hist = [hist[2:end];next_word]
        count+=1
    end

    average && return mean(total_loss)
    return (sum(total_loss), count) # if average is false
end

# Testing lossv1
@info "Testing loss_v1"
s = first(train_sentences)
avgloss = loss_v1(model,s)
(tot, cnt) = loss_v1(model, s, average = false)
@test 9 < avgloss < 10
@test cnt == length(s) + 1
@test tot/cnt ≈ avgloss

# Maploss function to operate in higher order loss functions
function maploss(lossfn, model, data; average = true)
    num_words = 0
    total_loss = []
    for sentence in data
        loss = lossfn(model,sentence, average = false)
        push!(total_loss, loss[1])
        num_words = num_words + loss[2]
    end

    average && return sum(total_loss)/num_words
    return (sum(total_loss), num_words)
end


# Testing maploss
@info "Testing maploss"
tst100 = collect(take(test_sentences, 100))
avgloss = maploss(loss_v1, model, tst100)
@test 9 < avgloss < 10
(tot, cnt) = maploss(loss_v1, model, tst100, average = false)
@test cnt == length(tst100) + sum(length.(tst100))
@test tot/cnt ≈ avgloss

# Timing loss for v1
@info "Timing loss_v1 with 1000 sentences"
tst1000 = collect(take(test_sentences, 1000))
GC.gc();@time maploss(loss_v1, model, tst1000)


# First Problem Occurs Here / Changing 100 to 10 works
@info "Timing loss_v1 training with 100 sentences"
GC.gc();trn100 = ((model,x) for x in collect(take(train_sentences, 100)))
GC.gc();@time progress(sgd!(loss_v1, trn100))  # Memory allocation issues force us to add these collectors

# Function to predict in parallel
function pred_v2(m::NNLM, hist::AbstractMatrix{Int})
    emb_inp = hist
    emb_out = m.embed(emb_inp)
    emb_out = dropout(reshape(emb_out,:,size(hist)[2]),m.dropout;seed=1)

    hid_inp =  emb_out
    hid_out = tanh.(m.hidden(hid_inp))
    hid_out = dropout(hid_out,m.dropout;seed=1)

    out = m.output(hid_out)

    return out
end

# Testing the predv2
@info "Testing pred_v2"

function scores_v2(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    return pred_v2(model, hist)
end


sent = first(test_sentences)
s1, s2 = scores_v1(model, sent), scores_v2(model, sent)
@test size(s1) == size(s2) == (length(train_vocab.i2w), length(sent)+1)
@test s1 ≈ s2

# Updated loss function for predv2
function loss_v2(m::NNLM, sent::AbstractVector{Int}; average = true)
    hist = [repeat([ model.vocab.eos ], model.windowsize); sent]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)

    average && return mean(nll(pred_v2(m,hist),[ sent; model.vocab.eos ]))
    return nll(pred_v2(m,hist),[ sent; model.vocab.eos ],average=false)

end

# Testing new loss function
@info "Testing loss_v2"
s = first(test_sentences)
@test loss_v1(model, s) ≈ loss_v2(model, s)
tst100 = collect(take(test_sentences, 100))
@test maploss(loss_v1, model, tst100) ≈ maploss(loss_v2, model, tst100)

# Timing for loss function v2
@info "Timing loss_v2  with 10K sentences"
tst10k = collect(take(train_sentences, 10000))
GC.gc();@time maploss(loss_v2, model, tst10k)


@info "Timing loss_v2 training with 1000 sentences"
trn1k = ((model,x) for x in collect(take(train_sentences, 1000)))
GC.gc();@time sgd!(loss_v2, trn1k)  # Memory allocation issues force us to add these collectors


# New format of the pred with batches
function pred_v3(m::NNLM, hist::Array{Int})
    emb_inp = hist[:,:,1]
    emb_out = m.embed(emb_inp)
    for entry in 2:size(hist)[ndims(hist)] # every entry
        curr_emb = m.embed(hist[:,:,entry])
        emb_out = cat(emb_out,curr_emb;dims=ndims(hist)+1)
    end

    emb_out = dropout(reshape(emb_out,:,size(hist)[2]*size(hist)[3]),m.dropout;seed=1)

    hid_inp =  emb_out
    hid_out = tanh.(m.hidden(hid_inp))
    hid_out = dropout(hid_out,m.dropout;seed=1)

    out = reshape(m.output(hid_out),:,size(hist)[2],size(hist)[3])
end

# Testing pred_v3
@info "Testing pred_v3"

function scores_v3(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    hist = reshape(hist, size(hist,1), 1, size(hist,2))
    return pred_v3(model, hist)
end

sent = first(train_sentences)
@test scores_v2(model, sent) ≈ scores_v3(model, sent)[:,1,:]

# Mask
function mask!(a,pad)
    row,col = size(a)
    col==1 && return a

    for i in 1:row
        for j in col:-1:2
            if a[i,(j-1)]==pad
                a[i,j]=0
            else
                break
            end
        end

    end
    return a
end

# Testing mask
@info "Testing mask!"
a = [1 2 1 1 1; 2 2 2 1 1; 1 1 2 2 2; 1 1 2 2 1]
@test mask!(a,1) == [1 2 1 0 0; 2 2 2 1 0; 1 1 2 2 2; 1 1 2 2 1]


##### We are here
# Loss v3
function loss_v3(m::NNLM, batch::AbstractMatrix{Int}; average = true)
    batch_size = first(size(batch))
    hist = hcat(repeat([ model.vocab.eos ], batch_size, model.windowsize), batch)

    long_length = size(hist)[2]-model.windowsize+1# length of the long sentence
    upd_hist = repeat([ model.vocab.eos ], model.windowsize , batch_size, long_length)

    for batch_order in 1:batch_size
        plane = vcat((hist[batch_order,i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
        upd_hist[:,batch_order,:] = reshape(plane, model.windowsize, 1, long_length)
    end


    average && return mean(nll(pred_v3(m,upd_hist),mask!(hcat(batch,repeat([model.vocab.eos],batch_size)),model.vocab.eos)))
    return nll(pred_v3(m,upd_hist),mask!(hcat(batch,repeat([model.vocab.eos],batch_size)),model.vocab.eos),average=false)
end

# Testing loss v3
@info "Testing loss_v3"
s = first(test_sentences)
b = [ s; model.vocab.eos ]'
@test loss_v2(model, s) ≈ loss_v3(model, b)

#######
# Minibatching
struct LMData
    src::TextReader
    batchsize::Int
    maxlength::Int
    bucketwidth::Int
    buckets
end

function LMData(src::TextReader; batchsize = 64, maxlength = typemax(Int), bucketwidth = 10)
    numbuckets = min(128, maxlength ÷ bucketwidth)
    buckets = [ [] for i in 1:numbuckets ]
    LMData(src, batchsize, maxlength, bucketwidth, buckets)
end

Base.IteratorSize(::Type{LMData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{LMData}) = Base.HasEltype()
Base.eltype(::Type{LMData}) = Matrix{Int}

function Base.iterate(d::LMData, state=nothing)
    if state == nothing
        for b in d.buckets; empty!(b); end
    end
    bucket,ibucket = nothing,nothing
    while true
        iter = (state === nothing ? iterate(d.src) : iterate(d.src, state))
        if iter === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent, state = iter
            if length(sent) > d.maxlength || length(sent) == 0; continue; end
            ibucket = min(1 + (length(sent)-1) ÷ d.bucketwidth, length(d.buckets))
            bucket = d.buckets[ibucket]
            push!(bucket, sent)
            if length(bucket) === d.batchsize; break; end
        end
    end
    if bucket === nothing; return nothing; end
    batchsize = length(bucket)
    maxlen = maximum(length.(bucket))
    batch = fill(d.src.vocab.eos, batchsize, maxlen + 1)
    for i in 1:batchsize
        batch[i, 1:length(bucket[i])] = bucket[i]
    end
    empty!(bucket)
    return batch, state
end

# ### Timing loss_v3
#
# We can compare the speeds of `loss_v2` and `loss_v3` using various batch sizes. Running
# the following on a V100 suggests that for forward loss calculation, a batchsize around 16
# gives the best speed.

@info "Timing loss_v2 and loss_v3 at various batch sizes"
@info loss_v2; test_collect = collect(test_sentences)
GC.gc(); @time p2 = maploss(loss_v2, model, test_collect)
for B in (1, 8, 16, 32, 64, 128, 256)
    @info loss_v3,B; test_batches = collect(LMData(test_sentences, batchsize = B))
    GC.gc(); @time p3 = maploss(loss_v3, model, test_batches); @test p3 ≈ p2
end

# For training, a batchsize around 64 seems best, although things are a bit more complicated
# here: larger batch sizes make fewer updates per epoch which may slow down convergence. We
# will use the smaller test data to get quick results.

@info "Timing SGD for loss_v2 and loss_v3 at various batch sizes"
train(loss, model, data) = sgd!(loss, ((model,sent) for sent in data))
@info loss_v2; test_collect = collect(test_sentences)
GC.gc(); @time train(loss_v2, model, test_collect)
for B in (1, 8, 16, 32, 64, 128, 256)
    @info loss_v3,B; test_batches = collect(LMData(test_sentences, batchsize = B))
    GC.gc(); @time train(loss_v3, model, test_batches)
end

# ## Part 7. Training
#
# You should be able to get the validation loss under 5.1 (perplexity under 165) in 100
# epochs with default parameters.  This takes about 5 minutes on a V100 GPU.
#
# Please review Knet function `progress!` and iterator function `ncycle` used below.

model = NNLM(train_vocab, HIST, EMBED, HIDDEN, DROPOUT)
train_batches = collect(LMData(train_sentences))
valid_batches = collect(LMData(valid_sentences))
test_batches = collect(LMData(test_sentences))
train_batches50 = train_batches[1:50] # Small sample for quick loss calculation

epoch = adam(loss_v3, ((model, batch) for batch in train_batches))
bestmodel, bestloss = deepcopy(model), maploss(loss_v3, model, valid_batches)

progress!(ncycle(epoch, 100), seconds=5) do x
    global bestmodel, bestloss
    ## Report gradient norm for the first batch
    f = @diff loss_v3(model, train_batches[1])
    gnorm = sqrt(sum(norm(grad(f,x))^2 for x in params(model)))
    ## Report training and validation loss
    trnloss = maploss(loss_v3, model, train_batches50)
    devloss = maploss(loss_v3, model, valid_batches)
    ## Save model that does best on validation data
    if devloss < bestloss
        bestmodel, bestloss = deepcopy(model), devloss
    end
    (trn=exp(trnloss), dev=exp(devloss), ∇=gnorm)
end


# Now you can generate some original sentences with your trained model:

## julia> generate(bestmodel)
## "the nasdaq composite index finished at N compared with ual earlier in the statement"
##
## julia> generate(bestmodel)
## "in the pentagon joseph r. waertsilae transactions the 1\\/2-year transaction was oversubscribed an analyst at <unk>"
