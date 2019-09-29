# initialize the Julia file naive-bow
using Pkg
Pkg.add("Knet")
# Pkg.add("Random")
# Pkg.add("HDF5")
# Pkg.add("JLD")

VOCAB_SIZE_THRESHOLD = 50 # represents most common words, which are less effective
VOCAB_SIZE = 1000
SENTENCE_SIZE = 150
#NUM_SENTENCES = 500
TRAIN_SIZE = 25000
TEST_SIZE = 25000
UNKNOWN = "<unk>"
# initialize the polarity matrix

# change this for a better approach
dir = "/home/minuteman/academics/'19 Fall/NLP/Project-Repo/NLP-Projects/aclImdb_v1/aclImdb/"
using Knet

# function pred(sentence_vector)
#     q_y = 1/2 # as it is provided
#     P = copy(q_y)
#     for i in 1:VOCAB_SIZE
#         if sentence_vector[i] != 0
#             for j in 1:sentence_vector[i]
#                 # P = P * polarity[i]
#             end
#         end
#     end
# end
#
# function readFiles(dir) # file is the one with .feat directory
#     data = []
#     train_dir = dir*"/train"
#
#     pos_dir = train_dir*"/pos"
#     for file_dir in readdir(pos_dir)
#         for line in eachline(pos_dir*"/"*file_dir)
#             sentence =  strip(lowercase(line))
#             word_ids = w2i.(split(sentence))
#             tag_id = t2i(file_dir[end-4]) # rate of the comment
#             push!(data,(word_ids,tag_id))
#         end
#     end
#
#     neg_dir = train_dir*"/neg"
#     for file_dir in readdir(neg_dir)
#         for line in eachline(neg_dir*"/"*file_dir)
#             sentence =  strip(lowercase(line))
#             word_ids = w2i.(split(sentence))
#             tag_id = t2i(file_dir[end-4]) # rate of the comment
#             push!(data,(word_ids,tag_id))
#         end
#     end
#     return data
# end



# declare the maximum likelihood function and the train loop
# function loss()

# wdict = Dict()
# tdict = Dict()
# w2i(x) = get!(wdict, x, 1+length(wdict))
# t2i(x) = get!(tdict, x, 1+length(tdict))
# UNK = w2i("<unk>")

# using HDF5, JLD
# d = readFiles(dir)
# cd(dir)
# save("data.jld","data",d)
# first(d)

# thanks https://www.rosettacode.org/wiki/Strip_a_set_of_characters_from_a_string#Julia for their kind advice
stripChar = (s, r) -> replace(s, Regex("[$r]") => " ")

function readandprep(dir)
    train_dir = dir*"/train"
    sentences =[]

    # first half is positive
    pos_dir = train_dir*"/pos"
    for file_dir in readdir(pos_dir)
        for line in eachline(pos_dir*"/"*file_dir)
            sentence = strip(lowercase(line))
            sentence = stripChar(sentence, """.,:!?#"'~/=-><""")
            sentence = split(sentence)
            tag_id = file_dir[end-4]
            if first(size(sentence))>SENTENCE_SIZE
                sentence = sentence[1:SENTENCE_SIZE]
            else
                while first(size(sentence)) < SENTENCE_SIZE
                    push!(sentence,UNKNOWN)
                end
            end
            push!(sentences,(sentence,tag_id)) # add those sentences to the all sentences matrix
        end
    end

    # second half is negative
    neg_dir = train_dir*"/neg"
    for file_dir in readdir(neg_dir)
        for line in eachline(neg_dir*"/"*file_dir)
            sentence = strip(lowercase(line))
            sentence = stripChar(sentence, """.,:!?#"'~/=-><""")
            sentence = split(sentence)
            tag_id = file_dir[end-4]
            if first(size(sentence))>SENTENCE_SIZE
                sentence = sentence[1:SENTENCE_SIZE]
            else
                while first(size(sentence)) < SENTENCE_SIZE
                    push!(sentence,UNKNOWN)
                end
            end
            push!(sentences,(sentence,tag_id))
        end
    end

    return sentences
end

function obtainvocab(sentences)
    vocab_freq = Dict{String,Int64}() # Pair array which holds vocab and freq pairs
    pos_freq = Dict{String,Int64}()
    neg_freq = Dict{String,Int64}()
    def_freq = 0

    for i in 1:TRAIN_SIZE # also equals to size(sentences)
        #print(first(i))
        if i == TRAIN_SIZE/2
            pos_freq = copy(vocab_freq)
        end
        for word in first(sentences[i]) # get the splitted sentence
            #println(word)
            if word != UNKNOWN
                vocab_freq[word] = get!(vocab_freq,word,def_freq) + 1 # 1 occurunce means, +1 in freq matrix
                if i > TRAIN_SIZE/2
                    neg_freq[word] = get!(neg_freq,word,def_freq) + 1
                end
            end
        end
    end


    vocab_freq = sort(collect(vocab_freq), by = tuple -> last(tuple), rev=true)[VOCAB_SIZE_THRESHOLD:VOCAB_SIZE_THRESHOLD+VOCAB_SIZE-2] # one room for unk

    pos_specific = Dict{String,Float32}()
    neg_specific = Dict{String,Float32}()
    voc_specific = Dict{String,Float32}()

    pos_sent_num_words = 0
    neg_sent_num_words = 0
    total_sent_num_words = 0

    for pair in vocab_freq
        word = pair[1]

        pos_f = get(pos_freq,word,0)
        pos_specific[word] = pos_f*1.0
        pos_sent_num_words += pos_f

        neg_f = get(neg_freq,word,0)
        neg_specific[word] = neg_f*1.0
        neg_sent_num_words += neg_f

        total_f = pair[2]
        voc_specific[word] = pair[2]*1.0 # again convert it to a dictionary
        total_sent_num_words+=total_f
    end

    for word in keys(voc_specific)
        pos_specific[word]/=pos_sent_num_words
        neg_specific[word]/=neg_sent_num_words
        voc_specific[word]/=total_sent_num_words
    end

    return pos_specific,neg_specific,voc_specific

    #return vocab_freq[1:VOCAB_SIZE]
end

sentence_tag_dict = readandprep(dir) # sentences and tags stored here
pos_freq_dict,neg_freq_dict,vocab_freq_dict = obtainvocab(sentence_tag_dict)



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
#
# println(pos_f)
# println(neg_f)
# println(total_f)

# after this step, negative and positive comments must be shuffled in order to eliminate overfitting
function prepwrtvocab(sentences, vocab)
    for sen_pos in 1:TRAIN_SIZE
        sentence = first(sentences[sen_pos])
        for word_pos in 1:SENTENCE_SIZE
            if !haskey(vocab,sentence[pos])
                sentence[word_pos] = UNKNOWN
            end
        end
        first(sentences[sen_pos])=sentence # update
    end

    return sentences
end

preped_sent = prepwrtvocab(sentence_tag_dict,vocab_freq_dict)

same = 0

for step in  1:TRAIN_SIZE
    if sentence_tag_dict[step]==preped_sent[step]
        global same+=1
    end
end

println(same*1.0/TRAIN_SIZE)
