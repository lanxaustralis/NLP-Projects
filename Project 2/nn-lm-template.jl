#jl Use `Literate.notebook(juliafile, ".", execute=false)` to convert to notebook.

# # A Neural Probabilistic Language Model
#
# Reference: Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *Journal of machine learning research, 3*. (Feb), 1137-1155. ([PDF](http://www.jmlr.org/papers/v3/bengio03a.html), [Sample code](https://github.com/neubig/nn4nlp-code/blob/master/02-lm))

using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging


# Set `datadir` to the location of ptb on your filesystem. You can find the ptb data in the
# https://github.com/neubig/nn4nlp-code repo
# [data](https://github.com/neubig/nn4nlp-code/tree/master/data) directory. The code below
# clones the nn4nlp-code repo using `git clone https://github.com/neubig/nn4nlp-code.git` if
# the data directory does not exist.

const datadir = "nn4nlp-code/data/ptb"
isdir(datadir) || run(`git clone https://github.com/neubig/nn4nlp-code.git`)


# ## Part 1. Vocabulary
#
# In this part we are going to implement a `Vocab` type that will map words to unique integers. The fields of `Vocab` are:
# * w2i: A dictionary from word strings to integers.
# * i2w: An array mapping integers to word strings.
# * unk: The integer id of the unknown word token.
# * eos: The integer id of the end of sentence token.
# * tokenizer: The function used to tokenize sentence strings.

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

# ### Vocab constructor
#
# Implement a constructor for the `Vocab` type. The constructor should take a file path as
# an argument and create a `Vocab` object with the most frequent words from that file and
# optionally unk and eos tokens. The keyword arguments are:
#
# * tokenizer: The function used to tokenize sentence strings.
# * vocabsize: Maximum number of words in the vocabulary.
# * mincount: Minimum count of words in the vocabulary.
# * unk, eos: unk and eos strings, should be part of the vocabulary unless set to nothing.
#
# You may find the following Julia functions useful: `Dict`, `eachline`, `split`, `get`,
# `delete!`, `sort!`, `keys`, `collect`, `push!`, `pushfirst!`, `findfirst`. You can take
# look at their documentation using e.g. `@doc eachline`.

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    ## Your code here
end

#-

@info "Testing Vocab"
f = "$datadir/train.txt"
v = Vocab(f)
@test all(v.w2i[w] == i for (i,w) in enumerate(v.i2w))
@test length(Vocab(f).i2w) == 10000
@test length(Vocab(f, vocabsize=1234).i2w) == 1234
@test length(Vocab(f, mincount=5).i2w) == 9859

# We will use the training data as our vocabulary source for the rest of the assignment. It
# has already been tokenized, lowercased, and words other than the most frequent 10000 have
# been replaced with `"<unk>"`.

train_vocab = Vocab("$datadir/train.txt")


# ## Part 2. TextReader
#
# Next we will implement `TextReader`, an iterator that reads sentences from a file and
# returns them as integer arrays using a `Vocab`.  We want to implement `TextReader` as an
# iterator for scalability. Instead of reading the whole file at once, `TextReader` will
# give us one sentence at a time as needed (similar to how `eachline` works). This will help
# us handle very large files in the future.

struct TextReader
    file::String
    vocab::Vocab
end

# ### iterate
#
# The main function to implement for a new iterator is `iterate`. The `iterate` function
# takes an iterator and optionally a state, and returns a `(nextitem,state)` if the iterator
# has more items or `nothing` otherwise. A one argument call `iterate(x)` starts the
# iteration, and a two argument call `iterate(x,state)` continues from where it left off.
#
# Here are some sources you may find useful on iterators:
#
# * https://github.com/denizyuret/Knet.jl/blob/master/tutorial/25.iterators.ipynb
# * https://docs.julialang.org/en/v1/manual/interfaces
# * https://docs.julialang.org/en/v1/base/collections/#lib-collections-iteration-1
# * https://docs.julialang.org/en/v1/base/iterators
# * https://docs.julialang.org/en/v1/manual/arrays/#Generator-Expressions-1
# * https://juliacollections.github.io/IterTools.jl/stable
#
# For `TextReader` the state should be an `IOStream` object obtained by `open(file)` at the
# start of the iteration. When `eof(state)` indicates that end of file is reached, the
# stream should be closed by `close(state)` and `nothing` should be returned. Otherwise
# `TextReader` reads the next line from the file using `readline`, tokenizes it, maps each
# word to its integer id using the vocabulary and returns the resulting integer array
# (without any eos tokens) and the state.

function Base.iterate(r::TextReader, s=nothing)
    ## Your code here
end

# These are some optional functions that can be defined for iterators. They are required for
# `collect` to work, which converts an iterator to a regular array.

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

#- 

@info "Testing TextReader"
train_sentences, valid_sentences, test_sentences =
    (TextReader("$datadir/$file.txt", train_vocab) for file in ("train","valid","test"))
@test length(first(train_sentences)) == 24
@test length(collect(train_sentences)) == 42068
@test length(collect(valid_sentences)) == 3370
@test length(collect(test_sentences)) == 3761


# ## Part 3. Model
#
# We are going to first implement some reusable layers for our model. Layers and models are
# basically functions with associated parameters. Please review [Function-like
# objects](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects-1) for how
# to best define such objects in Julia.

# ### Embed
#
# `Embed` is a layer that takes an integer or an array of integers as input, uses them as
# column indices to lookup embeddings in its parameter matrix `w`, and returns these columns
# packed into an array. If the input size is `(X1,X2,...)`, the output size will be
# `(C,X1,X2,...)` where C is the columns size of `w` (which Julia will automagically
# accomplish if you use the right indexing expression). Please review [Array
# indexing](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing-1) and the
# Knet `param` function to implement this layer.

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    ## Your code here
end

function (l::Embed)(x)
    ## Your code here
end

#-

@info "Testing Embed"
Knet.seed!(1)
embed = Embed(100,10)
input = rand(1:100, 2, 3)
output = embed(input)
@test size(output) == (10, 2, 3)
@test norm(output) ≈ 0.59804f0


# ### Linear
#
# The `Linear` layer implements an affine transformation of its input: `w*x .+ b`. `w`
# should be initialized with small random numbers and `b` with zeros. Please review `param`
# and `param0` functions from Knet for this.

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    ## Your code here
end

function (l::Linear)(x)
    ## Your code here
end

#-

@info "Testing Linear"
Knet.seed!(1)
linear = Linear(100,10)
input = oftype(linear.w, randn(Float32, 100, 5))
output = linear(input)
@test size(output) == (10, 5)
@test norm(output) ≈ 5.5301356f0


# ### NNLM
#
# `NNLM` is the model object. It has the following fields:
# * vocab: The `Vocab` object associated with this model.
# * windowsize: How many words of history the model looks at (ngram order).
# * embed: An `Embed` layer.
# * hidden: A `Linear` layer which should be followed by `tanh.` to produce the hidden activations.
# * output: A `Linear` layer to map hidden activations to vocabulary scores.
# * dropout: A number between 0 and 1 indicating dropout probability.

struct NNLM; vocab; windowsize; embed; hidden; output; dropout; end

# The constructor for `NNLM` takes a vocabulary and various size parameters, returns an
# `NNLM` object. Remember that the embeddings for `windowsize` words will be concatenated
# before being fed to the hidden layer.

function NNLM(vocab::Vocab, windowsize::Int, embedsize::Int, hiddensize::Int, dropout::Real)
    ## Your code here
end

#-

## Default model parameters
HIST = 3
EMBED = 128
HIDDEN = 128
DROPOUT = 0.5
VOCAB = length(train_vocab.i2w)

#-

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


# ## Part 4. One word at a time
#
# Conceptually the easiest way to implement the neural language model is by processing one
# word at a time. This is also computationally the most expensive, which we will address in
# upcoming parts.

# ### pred_v1
#
# `pred_v1` takes a model and a `windowsize` length vector of integer word ids indicating the
# current history, and returns a vocabulary sized vector of scores for the next word. The
# embeddings of the `windowsize` words are reshaped to a single vector before being fed to the
# hidden layer. The hidden output is passed through elementwise `tanh` before being fed to
# the output layer. Dropout is applied to embedding and hidden outputs.
#
# Please review Julia functions `vec`, `reshape`, `tanh`, and Knet function `dropout`.

function pred_v1(m::NNLM, hist::AbstractVector{Int})
    @assert length(hist) == m.windowsize
    ## Your code here
end

#-

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

sent = first(train_sentences)
@test size(scores_v1(model, sent)) == (length(train_vocab.i2w), length(sent)+1)

# ### generate
#
# `generate` takes a model `m` and generates a random sentence of maximum length
# `maxlength`. It initializes a history of `m.windowsize` `m.vocab.eos` tokens. Then it
# computes the scores for the next word using `pred_v1` and samples a next word using
# normalized exp of scores as probabilities. It pushes this next word into history and keeps
# going until `m.vocab.eos` is picked or `maxlength` is reached. It returns a sentence
# string consisting of concatenated word strings separated by spaces.
#
# Please review Julia functions `repeat`, `push!`, `join` and StatsBase function `sample`.

function generate(m::NNLM; maxlength=30)
    ## Your code here
end

#-

@info "Testing generate"
s = generate(model, maxlength=5)
@test s isa String
@test length(split(s)) <= 5


# ### loss_v1
#
# `loss_v1` computes the negative log likelihood loss given a model `m` and sentence `sent`
# using `pred_v1`. If `average=true` it returns the per-word average loss, if
# `average=false` it returns a `(total_loss, num_words)` pair. To compute the loss it starts
# with a history of `m.windowsize` `m.vocab.eos` tokens like `generate`. Then, for each word
# in `sent` and a final `eos` token, it computes the scores based on the history, converts
# them to negative log probabilities, adds the entry corresponding to the current word to
# the total loss and pushes the current word to history.
#
# Please review Julia functions `repeat`, `vcat` and Knet functions `logp`, `nll`.

function loss_v1(m::NNLM, sent::AbstractVector{Int}; average = true)
    ## Your code here
end

#-

@info "Testing loss_v1"
s = first(train_sentences)
avgloss = loss_v1(model,s)
(tot, cnt) = loss_v1(model, s, average = false)
@test 9 < avgloss < 10
@test cnt == length(s) + 1
@test tot/cnt ≈ avgloss


# ### maploss
#
# `maploss` takes a loss function `lossfn`, a model `model` and a dataset `data` and returns
# the average per word negative log likelihood loss if `average=true` or `(total_loss,num_words)` 
# if `average=false`. `data` may be an iterator over sentences (e.g. `TextReader`) or batches 
# of sentences. Computing the loss over a whole dataset is useful to monitor our performance 
# during training.

function maploss(lossfn, model, data; average = true)
    ## Your code here
end

#-

@info "Testing maploss"
tst100 = collect(take(test_sentences, 100))
avgloss = maploss(loss_v1, model, tst100)
@test 9 < avgloss < 10
(tot, cnt) = maploss(loss_v1, model, tst100, average = false)
@test cnt == length(tst100) + sum(length.(tst100))
@test tot/cnt ≈ avgloss

# ### Timing loss_v1
#
# Unfortunately processing data one word at a time is not very efficient. The following
# shows that we can only train about 40-50 sentences per second on a V100 GPU. The training
# data has 42068 sentences which would take about 1000 seconds or 15 minutes. We probably
# need 10-100 epochs for convergence which is getting too long for this assignment. Let's
# see if we can speed things up by processing more data in parallel.
#
# Please review Knet function `sgd!` used below as well as iterator functions `collect`,
# `take`, and [Generator
# expressions](https://docs.julialang.org/en/v1/manual/arrays/#Generator-Expressions-1).

@info "Timing loss_v1 with 1000 sentences"
tst1000 = collect(take(test_sentences, 1000))
@time maploss(loss_v1, model, tst1000)

@info "Timing loss_v1 training with 100 sentences"
trn100 = ((model,x) for x in collect(take(train_sentences, 100)))
@time sgd!(loss_v1, trn100)

# ## Part 5. One sentence at a time
#
# We may have to do things one word at a time when generating a sentence, but there is no
# reason not to do things in parallel for loss calculation. In this part you will implement
# `pred_v2` and `loss_v2` which do calculations for the whole sentence.

# ### pred_v2
#
# `pred_v2` takes a model `m`, an N×S array of word ids `hist` and produces a V×S array of
# scores where N is `m.windowsize`, V is the vocabulary size and `S` is sentence length
# including the final eos token. The `hist` array has already been padded and shifted such
# that `hist[:,i]` is the N word context to predict word i. `pred_v2` starts by finding the
# embeddings for all hist entries at once, a E×N×S array where E is the embedding size. The
# N embeddings for each context are concatenated by reshaping this array to (E*N)×S. After a
# dropout step, the hidden layer converts this to an H×S array where H is the hidden
# size. Following a `tanh` and `dropout`, the output layer produces the final result as a
# V×S array.

function pred_v2(m::NNLM, hist::AbstractMatrix{Int})
    ## Your code here
end

#-

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


# ### loss_v2
#
# `loss_v2` computes the negative log likelihood loss given a model `m` and sentence `sent`
# using `pred_v2`. If `average=true` it returns the per-word average loss, if
# `average=false` it returns a `(total_loss, num_words)` pair. To compute the loss it
# constructs a N×S history matrix such that `hist[:,i]` gives the N word context to predict
# word i where N is `m.windowsize` and S is the sentence length + 1 for the final eos token.
# Then it computes the scores for all S tokens using `pred_v2`, converts them to negative
# log probabilities, computes the loss based on the entries for the correct words.
#
# Please review the Knet function `nll`.

function loss_v2(m::NNLM, sent::AbstractVector{Int}; average = true)
    ## Your code here
end

#-

@info "Testing loss_v2"
s = first(test_sentences)
@test loss_v1(model, s) ≈ loss_v2(model, s)
tst100 = collect(take(test_sentences, 100))
@test maploss(loss_v1, model, tst100) ≈ maploss(loss_v2, model, tst100)


# ### Timing loss_v2
#
# The following tests show that loss_v2 works about 15-20 times faster than loss_v1 during
# maploss and training. We can train at 800+ sentences/second on a V100 GPU, which is under
# a minute per epoch. We could stop here and train a reasonable model, but let's see if we
# can squeeze a bit more performance by minibatching sentences.

@info "Timing loss_v2  with 10K sentences"
tst10k = collect(take(train_sentences, 10000))
@time maploss(loss_v2, model, tst10k)

@info "Timing loss_v2 training with 1000 sentences"
trn1k = ((model,x) for x in collect(take(train_sentences, 1000)))
@time sgd!(loss_v2, trn1k)


# ## Part 6. Multiple sentences at a time (minibatching)
#
# To get even more performance out of a GPU we will process multiple sentences at a
# time. This is called minibatching and is unfortunately complicated by the fact that the
# sentences in a batch may not be of the same length. Let's first write the minibatched
# versions of `pred` and `loss`, and see how to batch sentences together later.

# ### pred_v3
#
# `pred_v3` takes a model `m`, a N×B×S dimensional history array `hist`, and returns a V×B×S
# dimensional score array, where N is `m.windowsize`, V is the vocabulary size, B is the batch
# size, and S is maximum sentence length in the batch + 1 for the final eos token. First,
# the embeddings for all entries in `hist` are looked up, which results in an array of
# E×N×B×S where E is the embedding size. The embedding array is reshaped to (E*N)×(B*S) and
# dropout is applied. It is then fed to the hidden layer which returns a H×(B*S) hidden
# output where H is the hidden size. Following element-wise tanh and dropout, the output
# layer turns this into a score array of V×(B*S) which is reshaped and returned as a V×B×S
# dimensional tensor.

function pred_v3(m::NNLM, hist::Array{Int})
    ## Your code here
end

#-

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


# ### mask!
#
# `mask!` takes matrix `a` and a pad value `pad`. It replaces all but one of the pads at the
# end of each row with 0's. This can be used in `loss_v3` for the loss calculation: the Knet
# `nll` function skips 0's in the answer array.

function mask!(a,pad)
    ## Your code here
end

#-

@info "Testing mask!"
a = [1 2 1 1 1; 2 2 2 1 1; 1 1 2 2 2; 1 1 2 2 1]
@test mask!(a,1) == [1 2 1 0 0; 2 2 2 1 0; 1 1 2 2 2; 1 1 2 2 1]


# ### loss_v3
#
# `loss_v3` computes the negative log likelihood loss given a model `m` and sentence
# minibatch `batch` using `pred_v3`. If `average=true` it returns the per-word average loss,
# if `average=false` it returns a `(total_loss, num_words)` pair. The batch array has
# dimensions B×S where B is the batch size and S is the length of the longest sentence in
# the batch + 1 for the final eos token. Each row contains the word ids of a sentence padded
# with eos tokens on the right.  Sentences in a batch may have different lengths. `loss_v3`
# first constructs a history array of size N×B×S from the batch such that `hist[:,i,j]`
# gives the N word context to the j'th word of the i'th sentence. This is done by repeating,
# slicing, concatenating, reshaping and/or using permutedims on the batch array. Next
# `pred_v3` is used to compute the scores array of size V×B×S where V is the vocabulary
# size. The correct answers are extracted from the batch to an array of size B×S and the
# extra padding at the end of each sentence (after the final eos) is masked (extra eos
# replaced by zeros).  Finally the scores and the masked correct answers are used to compute
# the negative log likelihood loss using `nll`.
#
# Please review array slicing, Julia functions `vcat`, `hcat`, `reshape`, `permutedims`, and
# the Knet function `nll` for this exercise.

function loss_v3(m::NNLM, batch::AbstractMatrix{Int}; average = true)
    ## Your code here
end

#-

@info "Testing loss_v3"
s = first(test_sentences)
b = [ s; model.vocab.eos ]'
@test loss_v2(model, s) ≈ loss_v3(model, b)


# ### Minibatching
#
# Below is a sample implementation of a sequence minibatcher. The `LMData` iterator wraps a
# TextReader and produces batches of sentences with similar length to minimize padding (too
# much padding wastes computation). To be able to scale to very large files, we do not want
# to read the whole file, sort by length etc. Instead `LMData` keeps around a small number
# of buckets and fills them with similar sized sentences from the TextReader. As soon as one
# of the buckets reaches the desired batch size it is turned into a matrix with the
# necessary padding and output. When the TextReader is exhausted the remaining buckets are
# returned (which may have smaller batch sizes). I will let you figure the rest out from the
# following, there is no code to write for this part.

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
