
using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    ## Your code here
    open(file) do f
        w2i = Dict{String, Int64}(unk => 1, eos => 2)
        i2w = Vector{String}()
        push!(i2w, unk)
        push!(i2w, eos)
        
        for line in eachline(f)
            sentence = tokenizer(line, ['.',' '], keepempty = false)

            for word in sentence
                ind = (get!(w2i, word, 1+length(w2i)))
                if (length(i2w) < ind)
                    push!(i2w, word)
                end
            end
        end
        close(f)
        return Vocab(w2i, i2w, 1, 2, tokenizer)
    end
end

file = "test_text.txt"
v = Vocab(file)

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    ## Your code here
    return Embed(param(vocabsize,embedsize))
end

function (l::Embed)(x)
    ## Your code here
    W = l.w
    sx = size(x)
    L = Array{Float64}(undef, size(W)[2],sx[1],sx[2])
    for i in range(1,sx[1])
        for j in range(1,sx[2])
            L[:,i,j] = W[x[i,j],:]
        end
    end
    return L 
end

Knet.seed!(1)
embed = Embed(10,10)
input = rand(1:10, 2, 3)
output = embed(input)

output
@test size(output) == (10, 2, 3)
#@test norm(output) ≈ 0.59804f0

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    ## Your code here
    w = param(outputsize,inputsize)
    b = param0(outputsize)
    return Linear(w,b)
end

function (l::Linear)(x)
    ## Your code here
    l.w * x .+ l.b
end

@info "Testing Linear"
Knet.seed!(1)
linear = Linear(100,10)
input = oftype(linear.w, randn(Float32, 100, 5))
output = linear(input)
@test size(output) == (10, 5)
@test norm(output) ≈ 5.5301356f0

struct NNLM; vocab; windowsize; embed; hidden; output; dropout; end

# The constructor for `NNLM` takes a vocabulary and various size parameters, returns an
# `NNLM` object. Remember that the embeddings for `windowsize` words will be concatenated
# before being fed to the hidden layer.

function NNLM(vocab::Vocab, windowsize::Int, embedsize::Int, hiddensize::Int, dropout::Real)
    ## Your code here
    vocabsize = length(v.i2w)
    ## NNLM (vocab::Vocab,  windowsize::Int,  embed::Embed,  hidden::Linear,  output::Linear,  dropout::Int)
    NNLM(vocab, windowsize, Embed(embedsize,vocabsize),Linear(embedsize*windowsize,hiddensize),Linear(hiddensize,vocabsize),dropout)
end

#-

## Default model parameters
HIST = 3
EMBED = 128
HIDDEN = 128
DROPOUT = 0.5
train_vocab = v
VOCAB = length(v.i2w)

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

@doc vec
