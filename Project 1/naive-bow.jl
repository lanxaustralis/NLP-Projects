# initialize the Julia file naive-bow
using Pkg
Pkg.add("Knet")
Pkg.add("Random")
# Pkg.add("HDF5")
# Pkg.add("JLD")
import Knet: accuracy


VOCAB_THRESHOLD = 20  # represents most common words, which are not class specific -> Increasing this constant is inefficient in terms of computational perform, however may a slight posiitive effect on the accuracy
VOCAB_SIZE = 50000     # gets most common words in order
SENTENCE_SIZE = 250 # increasing this constant, increases the accuracy value, in most cases.
TR_TS_SIZE = 25000
UNKNOWN = "<unk>"

using Knet: accuracy, param ,Param, @diff, grad, params, KnetArray
using Random

# thanks https://www.rosettacode.org/wiki/Strip_a_set_of_characters_from_a_string#Julia for their kind advice
stripChar = (s, r) -> replace(s, Regex("[$r]") => " ")

# change this directory wrt test environment
dir = "/home/minuteman/academics/'19 Fall/NLP/Project-Repo/NLP-Projects/aclImdb_v1/aclImdb/"

println("---Welcome---")
println("|- The constants of the program as follows -|\n")
println((Vocabulary_Size=VOCAB_SIZE,Train_Test_Size=TR_TS_SIZE,Unknown_Tag=UNKNOWN,Sentence_Size=SENTENCE_SIZE,Vocabulary_Commonality_Threshold=VOCAB_THRESHOLD))
function readandprep(dir)
    sentences = []

    excluded_chars = """.,:!?#"'~/=-><"""
    # first half is positive
    pos_dir = dir * "/pos"
    for file_dir in readdir(pos_dir)
        for line in eachline(pos_dir * "/" * file_dir)
            sentence = strip(lowercase(line))
            sentence = stripChar(sentence, excluded_chars)
            sentence = split(sentence)
            tag_id = file_dir[end-4]
            if first(size(sentence)) > SENTENCE_SIZE
                sentence = sentence[1:SENTENCE_SIZE]
            else
                while first(size(sentence)) < SENTENCE_SIZE
                    push!(sentence, UNKNOWN)
                end
            end
            push!(sentences, (sentence, tag_id)) # add those sentences to the all sentences matrix
        end
    end

    # second half is negative
    neg_dir = dir * "/neg"
    for file_dir in readdir(neg_dir)
        for line in eachline(neg_dir * "/" * file_dir)
            sentence = strip(lowercase(line))
            sentence = stripChar(sentence, excluded_chars)
            sentence = split(sentence)
            tag_id = file_dir[end-4]
            if first(size(sentence)) > SENTENCE_SIZE
                sentence = sentence[1:SENTENCE_SIZE]
            else
                while first(size(sentence)) < SENTENCE_SIZE
                    push!(sentence, UNKNOWN)
                end
            end
            push!(sentences, (sentence, tag_id))
        end
    end

    # do not worry, they will be all shuffled, soon :)
    return sentences
end

function obtainvocab(sentences)
    vocab_freq = Dict{String,Int64}() # Pair array which holds vocab and freq pairs
    pos_freq = Dict{String,Int64}()
    neg_freq = Dict{String,Int64}()
    def_freq = 1 # Laplace Smoothing in a smooth form :)

    for i = 1:TR_TS_SIZE # also equals to size(sentences)
        if i == TR_TS_SIZE / 2
            pos_freq = copy(vocab_freq)
        end
        for word in first(sentences[i]) # get the splitted sentence
            if word != UNKNOWN
                vocab_freq[word] = get!(vocab_freq, word, def_freq) + 1 # 1 occurunce means, +1 in freq matrix
                if i > TR_TS_SIZE / 2
                    neg_freq[word] = get!(neg_freq, word, def_freq) + 1
                end
            end
        end
    end


    vocab_freq = sort(
        collect(vocab_freq),
        by = tuple -> last(tuple),
        rev = true,
    )[VOCAB_THRESHOLD:VOCAB_THRESHOLD+VOCAB_SIZE-1]

    pos_specific = Dict{String,Float32}()
    neg_specific = Dict{String,Float32}()
    voc_specific = Dict{String,Float32}()

    # Latter approach
    pos_latter_d = rand(Float32,1,VOCAB_SIZE+1)
    neg_latter_d = rand(Float32,1,VOCAB_SIZE+1)
    voc_latter_d = Dict()
    w2i(x) = get!(voc_latter_d, x, 1+length(voc_latter_d))
    #tdict = Dict()
    #t2i(x) = get!(tdict, x, 1+length(tdict))
    UNK = w2i("<unk>")


    # Calculate probs for given sets
    pos_sent_num_words = 0
    neg_sent_num_words = 0
    total_sent_num_words = 0

    for pair in vocab_freq
        word = pair[1]

        pos_f = get(pos_freq, word, def_freq)
        pos_specific[word] = pos_f * 1.0
        pos_sent_num_words += pos_f

        neg_f = get(neg_freq, word, def_freq)
        neg_specific[word] = neg_f * 1.0
        neg_sent_num_words += neg_f

        total_f = pair[2]
        voc_specific[word] = pair[2] * 1.0 # again convert it to a dictionary
        total_sent_num_words += total_f

        # latter approach
        w2i(word) # added to the new dictionary
    end

    for word in keys(voc_specific)
        if word==UNKNOWN
            pos_latter_d[voc_latter_d[word]] = 1.0#/total_sent_num_words * total_sent_num_words
            neg_latter_d[voc_latter_d[word]] = 1.0#/total_sent_num_words * total_sent_num_words
            continue
        end

        pos_specific[word] /= pos_sent_num_words
        neg_specific[word] /= neg_sent_num_words
        voc_specific[word] /= total_sent_num_words

        # latter approach
        pos_latter_d[voc_latter_d[word]]=pos_specific[word] / voc_specific[word]
        neg_latter_d[voc_latter_d[word]]=neg_specific[word] / voc_specific[word]
    end

    return pos_specific,
        neg_specific,
        voc_specific,

        pos_sent_num_words,
        neg_sent_num_words,
        total_sent_num_words,

        pos_latter_d,
        neg_latter_d,
        voc_latter_d

end

println()
print("Preparation for training data ->")
@time train_word_tag = readandprep(dir * "/train") # sentences and tags stored here
print("Preparation for vocabulary ->")
@time pos_freq_dict,
    neg_freq_dict,
    vocab_freq_dict,

    pos_word_num,
    neg_word_num,
    total_word_num,

    pos_latter,
    neg_latter,
    voc_latter = obtainvocab(train_word_tag)
print("Preparation for test data ->")
@time test_word_tag = readandprep(dir * "/test")

# Please comment out here to check total probability of words within a set (i.e. vocabulary)
# Results must approximately equal to 1
#
# pos_f = 0.0
# neg_f = 0.0
# total_f = 0.0
#
# for words in keys(vocab_freq_dict)
#     global pos_f += pos_freq_dict[words]
#     global neg_f += neg_freq_dict[words]
#     global total_f += vocab_freq_dict[words]
# end
# println("The sum of probabilities of words within each classes, expected as ~1 as a result of the float point error")
# println("Positive Class -> "*string(pos_f))
# println("Negative Class -> "*string(neg_f))
# println("Total Trained Data -> "*string(total_f))
# println("----------------------------------------")

# after this step, negative and positive comments must be shuffled in order to eliminate overfitting
function prepwrtvocab(sentence_arr, vocab_dict)
    # There are two approaches below, one identification provided other is vanilla
    for i = 1:TR_TS_SIZE
        words, tag = sentence_arr[i]
        for j = 1:SENTENCE_SIZE
            if !haskey(vocab_dict, words[j])
                words[j] = UNKNOWN
            end
        end
        sentence_arr[i] = words, tag
    end

    return sentence_arr
end


# At the following code, it can be easily seen that the preprocessing is done for many sentences
# decreasing the VOCAB_THRESHOLD may effect the similarity ratio positively
# same = 0
#
# for step in  1:TR_TS_SIZE
#     if sentence_tag_dict[step]==preped_trn[step]
#         global same+=1
#     end
# end
#
# println("The similarity ratio of preprocessed and raw comments (decrease VOCAB_THRESHOLD to work on more similar sets) -> $(same*1.0/TR_TS_SIZE)")
# println("----------------------------------------")

function pred(sentence)
    p_vector = prob_p, prob_n = (1.0, 1.0)
    #println("Sentence is -> "*string(sentence))
    for word in sentence
        con_vector = pos_p, neg_p = calcwithLaplace(word)
        #println("Calculated vector for "*string(word)*" -> "*string(con_vector))
        p_vector = p_vector .* con_vector
        #println("Updated prob vector -> "*string(p_vector))
    end
    #println("Result p vector -> "*string(p_vector))
    return p_vector[1] > p_vector[2] # true for positive, false for negative
end

function calcwithLaplace(word)
    global pos_word_num
    global neg_word_num
    global total_word_num

    global vocab_freq_dict
    global pos_freq_dict
    global neg_freq_dict

# since the floating numbers becomes to low, it is applied the original format of the formula
    if !haskey(vocab_freq_dict, word)
        #println(word*" is a new word!!!")
        return (1.0 / pos_word_num, 1.0 / neg_word_num) .* total_word_num # it is already added the size of the vocab
    end
    return (pos_freq_dict[word], neg_freq_dict[word]) ./ vocab_freq_dict[word] # Laplace smoothing was set before
end

function predall(comment_tag_set)
    correct = 0
    #print("Valid ones ")
    for i = 1:TR_TS_SIZE
        sentence, tag = comment_tag_set[i]
        valid = pred(sentence) == tagclassifier(tag)
        if valid
            correct += 1
        end
    end
    return correct * 1.0 / TR_TS_SIZE
end

function tagclassifier(tag)
    return parse(Int64, tag) > 5 # tag > 5 are positive comments which result true, vice versa
end


# Pure Way - w/o explicit traininig

# Comment out below code
println()
print("Preprocessing train data ->")
preped_trn = @time prepwrtvocab(train_word_tag, vocab_freq_dict)
print("Preprocessing test data ->")
preped_tst = @time prepwrtvocab(test_word_tag, vocab_freq_dict)

rng = MersenneTwister(12345)
shuffle!(rng, preped_trn)
shuffle!(rng, preped_tst)

println()
print("Prediction for train->")
@time acc_trn = predall(preped_trn)
print("Prediction for test->")
@time acc_tst = predall(preped_tst)

println()
println("Accuracy for train -> " * string(acc_trn))
println("Accuracy for test -> " * string(acc_tst))


# Active Learning

# SGD seems not working for this case, therefore above code recomended for the performance monitoring

# But the matrix approach seems more efficent in terms of computational performance
# For the following strategy, one Weight matrix which includes conditional probabilities of
# each word with respect to given classes (size of 2*VOCAB_SIZE)
# Each word tokenized as Prof. Yuret recomended, which stored in a word -> id vocab
# which makes matrix operation easier as the sentence array turnd into id array and tag

# From now on, the strategy is changed to the on learning process

# First check container whether updates are correct or not
# function testStrategyChange()
#     global voc_latter
#
#     # for positive class update
#     global pos_freq_dict
#     global pos_latter
#
#     mismatch = 0
#
#     for (word,freq) in pos_freq_dict
#         if !haskey(voc_latter,word)
#             println(word*" is not registered on the newer dictionary")
#             mismatch+=1
#             continue
#         end
#         index = voc_latter[word]
#         if freq!=pos_latter[index]
#             println(word*" -> values of freq for this word does not match -- positive class")
#             mismatch+=1
#             #return
#         end
#     end
#
#     # for negative class update
#     global neg_freq_dict
#     global neg_latter
#
#     trial = VOCAB_SIZE
#     for (word,freq) in neg_freq_dict
#         if !haskey(voc_latter,word)
#             println(word*" is not registered on the newer dictionary")
#             mismatch+=1
#             continue
#         end
#         index = voc_latter[word]
#         if freq!=neg_latter[index]
#             println(word*" -> values of freq for this word does not match -- negative class")
#             mismatch+=1
#             #return
#         end
#     end
#
#     println("Total mismatch number over vocabulary size "*string(VOCAB_SIZE)*" -> "*string(mismatch))
# end

# Run this code to see whether there is any mismatch between both strategies
#testStrategyChange()


# Factorizaton for newer strategy
function prepwrtvocab(sentence_arr)
    global voc_latter
    sent_id_arr = []
    # There are two approaches below, one identification provided other is vanilla
    for i = 1:TR_TS_SIZE
        one_sent_id_arr = Array{Int64,1}()
        words, tag = sentence_arr[i]
        for j = 1:SENTENCE_SIZE
            if !haskey(voc_latter, words[j])
                push!(one_sent_id_arr,1) # unk
                continue
            end
            push!(one_sent_id_arr,voc_latter[words[j]])
        end
        push!(sent_id_arr,(one_sent_id_arr,tagclassifier(tag))) # true for positive, false for neg
    end

    return sent_id_arr # id tag array
end
println("----------------------------------------")
println("Matrix Strategy Results")
println()
# Contianers of new format
print("Preprocessing train data for matrix strategy ->")
preped_trn_latter = @time prepwrtvocab(train_word_tag)
print("Preprocessing test data for matrix strategy ->")
preped_tst_latter = @time prepwrtvocab(test_word_tag)
println()

# using Knet: param
# Knet.param(dims...) = Param(Array(0.01f0 * randn(Float32, dims...)))

W = Param(Array(vcat(pos_latter,neg_latter)))

pred(words) = prod(W[:,words], dims=2)

#  No need for this if we do not perform an optimization algorithm
# function loss(words, tag)
#     scores = pred(words)
#     logprobs = scores .- log(sum(exp.(scores)))
#     -logprobs[tag]
# end


function Knet.accuracy(data)
    return sum(argmax(pred(x))[1] == y for (x,y) in data) / TR_TS_SIZE
end


function train(; nepochs = 3, lr = 0.1) # epoch as
    for epoch in 1:nepochs
        shuffle!(preped_trn_latter)
        # SGD seems ineffective
        for (x,y) in preped_trn_latter
            tag = y ? 1 : 2 # if it is true, the it is in class 1 which is positive class
            ∇loss = @diff loss(x,tag)
            for p in params(∇loss)
                p .= p - lr * grad(∇loss, p)
            end
        end
        println((epoch = epoch, trn = accuracy(preped_trn_latter)))
    end
end

train()
println()
print("Result for test -> ")
println(accuracy(preped_tst_latter))
println("---End---")
