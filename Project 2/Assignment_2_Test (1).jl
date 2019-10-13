
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

struct TextReader
    file::String
    vocab::Vocab
end

word2ind(dict,x) = get(dict, x, 1)

function Base.iterate(r::TextReader, s=nothing)
    ## Your code here
    if s == nothing
        state = open(r.file)
        r,state
    else
        if eof(s) == true 
            close(s)
            return nothing
        else
            line = readline(s)
            
            sent = r.vocab.tokenizer(line, ['.',' '], keepempty = false)
            sent_ind = []
            for word in sent
                ind = word2ind(r.vocab.w2i,word)
                push!(sent_ind,ind)
            end
            return sent_ind, s
        end
    end
end

# These are some optional functions that can be defined for iterators. They are required for
# `collect` to work, which converts an iterator to a regular array.

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

#- 

@info "Testing TextReader"
train_sentences = (TextReader("test_text.txt", v))
@test length(first(train_sentences)) == 24
@test length(collect(train_sentences)) == 42068
@test length(collect(valid_sentences)) == 3370
@test length(collect(test_sentences)) == 3761

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    ## Your code here
    return Embed(param(vocabsize,embedsize))
end

function (l::Embed)(x)
    ## Your code here
    W = l.w
    sx = size(x)
    L = Array{Float64}(undef, size(W)[1],sx[1],sx[2])
    for i in range(1,stop=sx[1])
        for j in range(1,stop=sx[2])
            L[:,i,j] = W[:,x[i,j]]
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
@test norm(output) ≈ 0.59804f0

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

function pred_v1(m::NNLM, hist::AbstractVector{Int})
    @assert length(hist) == m.windowsize
    ## Your code here
    emb_inp = reshape(hist,m.windowsize,1)
    emb = dropout(m.embed,m.dropout;seed=1)
    emb_out = emb(emb_inp)

    hid_inp = vec(emb_out)
    hid = dropout(m.hidden,m.dropout;seed=1)
    hid_out = tanh.(hid(hid_inp))

    out = m.output(hid_out)

    return out
end

#

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

using Statistics
function loss_v1(m::NNLM, sent::AbstractVector{Int}; average = true)
    ## Your code here
    hist = repeat([ model.vocab.eos ], model.windowsize)
    total_loss = []
    for next_word in [ sent; model.vocab.eos ]
        s = pred_v1(model, hist)
        push!(total_loss, nll(s, next_word))
        hist = [hist[2:end];next_word]
    end
    
    if average
        return mean(total_loss)
    else
        return (total_loss, length(sent))
    end
end

@info "Testing loss_v1"
s = first(train_sentences)
avgloss = loss_v1(model,s)
(tot, cnt) = loss_v1(model, s, average = false)
@test 9 < avgloss < 10
@test cnt == length(s) + 1
@test tot/cnt ≈ avgloss

function maploss(lossfn, model, data; average = true)
    ## Your code here
    num_words = 0
    total_loss = []
    for sentence in data
        loss = lossfn(sentence, average = False)
        push!(total_loss, loss[1])
        num_words = num_words + loss[2]
    end
    
    if average
        return mean(total_loss)
    else
        return (total_loss, num_words)
    end
end

#-

@info "Testing maploss"
tst100 = collect(take(test_sentences, 100))
avgloss = maploss(loss_v1, model, tst100)
@test 9 < avgloss < 10
(tot, cnt) = maploss(loss_v1, model, tst100, average = false)
@test cnt == length(tst100) + sum(length.(tst100))
@test tot/cnt ≈ avgloss
