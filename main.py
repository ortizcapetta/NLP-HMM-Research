import nltk
from nltk.corpus import brown
tagged_words = [ ]

tagged_sents = brown.tagged_sents()
#tagset=universal changes the brown tags for the universal ones
#this is only for simplicity and better understanding but could be changed later on
'''The universal tagset consists of the follow PoS: 

ADJ 	adjective 	new, good, high, special, big, local
ADP 	adposition 	on, of, at, with, by, into, under
ADV 	adverb 	really, already, still, early, now
CONJ 	conjunction 	and, or, but, if, while, although
DET 	determiner, article 	the, a, some, most, every, no, which
NOUN 	noun 	year, home, costs, time, Africa
NUM 	numeral 	twenty-four, fourth, 1991, 14:24
PRT 	particle 	at, on, out, over per, that, up, with
PRON 	pronoun 	he, their, her, its, my, I, us
VERB 	verb 	is, say, told, given, playing, would
. 	punctuation marks 	. , ; !
X 	other 	ersatz, esprit, dunno, gr8, univeristy'''



'''brown corpus has words collected as tuples of (word,tag)
to utilize the nltk probabilistic packages we need them to be (tag,word)
so we can obtain P(tag|word), therefor we reverse them.
We utilize brown.tagged_sents instead of brown.tagged_words because we need to know
the start of the sentence to calculate the initial probability.'''

#iterate through sentences and through words to reverse them and set the Start probability
for sents in tagged_sents:
    tagged_words.append(("S","")) # S = start of sentence
    tagged_words.extend([(tag[:2],word) for (word,tag) in sents])



'''Emission Probability '''

#obtain frequency distribution utilizing the tags as conditions
cond_freqwords = nltk.ConditionalFreqDist(tagged_words)

#print(cond_freqwords.conditions())
#obtain conditional frequency
#P = P(tag|word))
condprob_words = nltk.ConditionalProbDist(cond_freqwords, nltk.MLEProbDist) #MLEProbDist = Maximum Likelyhood Estimate
print(condprob_words)

tags = []

#put all tags in an array
for x in tagged_words:
    tags.append(x[0])

'''Transition Probability'''

#utilize bigrams to distribute (the probablity that x tag would come after y tag) P(tag|tag(i-1)
cond_freqtags = nltk.ConditionalFreqDist(nltk.bigrams(tags))
condprob_tags = nltk.ConditionalProbDist(cond_freqtags, nltk.MLEProbDist)


distinct_tagsets = set(tags)

#distinct_tagsets = ["ADJ","ADP","ADV","CONJ","DET","CONJ","NOUN","NUM","PRT","PRON","VERB"] #excluding S


sample_sentence = "I saw the door open." \
                  ""
print(sample_sentence)
tokenized_sentence = nltk.word_tokenize(sample_sentence)
print(tokenized_sentence)

viterbi = [] #structure to hold the viterbi iterations for each word

probabilityStart={}


#handle the initial probablity of the first word in the sentence
for tag in distinct_tagsets:
   ''' result = P(Tag|'S') * P(Word|Tag) '''
   result = condprob_tags['S'].prob(tag) * condprob_words[tag].prob(tokenized_sentence[0])
   probabilityStart[tag] = result



bestValue = -1

for x in probabilityStart.keys():
    if(probabilityStart[x] > bestValue):
        bestTag = x

viterbi.append(probabilityStart)


for x in range (1,len(tokenized_sentence)):
    curr = {}
    bestRes = -1
    bestTag = ''
    for tag in distinct_tagsets:
        for prevtag in distinct_tagsets:
            '''result = P(Word|Tag) * P(Previous state probability) * P(tag|tag-1)
            This is to find all results of all possible tags in the state to find the best one'''
            result = condprob_words[tag].prob(tokenized_sentence[x]) * viterbi[x-1][prevtag] * condprob_tags[prevtag].prob(tag)
            if result > bestRes:
                bestRes = result
                bestTag = prevtag
            '''Calculates the best once having obtained all results'''
        curr[tag] = viterbi[x-1][bestTag] * condprob_tags[bestTag].prob(tag) * condprob_words[tag].prob(tokenized_sentence[x])

    viterbi.append(curr) #appends the calculation for that state



taglist = []
for x in viterbi:
    biggest = -1
    biggestTag = ''
    for y in distinct_tagsets:
        if(x[y] > biggest):
            biggest = x[y]
            biggestTag = y
    taglist.append(biggestTag)
print(taglist)
#print(viterbi)