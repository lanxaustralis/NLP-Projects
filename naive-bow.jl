# initialize the Julia file naive-bow
using Pkg
Pkg.add("Knet")
Pkg.add("Random")
# Pkg.add("HDF5")
# Pkg.add("JLD")

VOCAB_THRESHOLD = 15  # represents most common words, which are not class specific -> Increasing this constant is inefficient in terms of computational perform, however may a slight posiitive effect on the accuracy
VOCAB_SIZE = 30000
SENTENCE_SIZE = 300 # increasing this constant, increases the accuracy value, in most cases.
TR_TS_SIZE = 25000
UNKNOWN = "<unk>"

# change this for a better approach
dir = "/home/minuteman/academics/'19 Fall/NLP/Project-Repo/NLP-Projects/aclImdb_v1/aclImdb/"
using Knet

# thanks https://www.rosettacode.org/wiki/Strip_a_set_of_characters_from_a_string#Julia for their kind advice
stripChar = (s, r) -> replace(s, Regex("[$r]") => " ")

function readandprep(dir)
    sentences = []

    # first half is positive
    pos_dir = dir * "/pos"
    for file_dir in readdir(pos_dir)
        for line in eachline(pos_dir * "/" * file_dir)
            sentence = strip(lowercase(line))
            sentence = stripChar(sentence, """.,:!?#"'~/=-><""")
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
            sentence = stripChar(sentence, """.,:!?#"'~/=-><""")
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
    )[VOCAB_THRESHOLD:VOCAB_SIZE_THRESHOLD+VOCAB_SIZE-1]

    pos_specific = Dict{String,Float32}()
    neg_specific = Dict{String,Float32}()
    voc_specific = Dict{String,Float32}()

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
    end

    for word in keys(voc_specific)
        pos_specific[word] /= pos_sent_num_words
        neg_specific[word] /= neg_sent_num_words
        voc_specific[word] /= total_sent_num_words
    end

    return pos_specific,
        neg_specific,
        voc_specific,
        pos_sent_num_words,
        pos_sent_num_words,
        neg_sent_num_words,
        total_sent_num_words

    #return vocab_freq[1:VOCAB_SIZE]
end

@time train_word_tag = readandprep(dir * "/train") # sentences and tags stored here
@time pos_freq_dict,
    neg_freq_dict,
    vocab_freq_dict,
    pos_word_num,
    neg_word_num,
    total_word_num = obtainvocab(train_word_tag)

@time test_word_tag = readandprep(dir * "/test")

# Please comment out here to check total probability of words within a set (i.e. vocabulary)
# Results must converge to 1
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
        #print(sentence*string(tag))
        valid = pred(sentence) == tagclassifier(tag)
        if valid
            correct += 1
            #print(string(i)*" ")
        end
    end
    #println("")
    return correct * 1.0 / TR_TS_SIZE
end

function tagclassifier(tag)
    return parse(Int64, tag) > 5 # tag > 5 are positive comments which result true, vice versa
end


preped_trn = @time prepwrtvocab(train_word_tag, vocab_freq_dict)
preped_tst = @time prepwrtvocab(test_word_tag, vocab_freq_dict)

using Random
rng = MersenneTwister(12345)
shuffle!(rng, preped_trn)
shuffle!(rng, preped_tst)

@time acc_trn = predall(preped_trn)
@time acc_tst = predall(preped_tst)

println("Accuracy for train -> " * string(acc_trn))
println("Accuracy for test -> " * string(acc_tst))
