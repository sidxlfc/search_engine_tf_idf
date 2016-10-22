"""
Here is a summary of what the functions do :

"""

import os
from nltk.tokenize import RegexpTokenizer

import sys
from nltk.stem.porter import PorterStemmer
from collections import Counter

#getting the stopwords ready
from nltk.corpus import stopwords
stopwords_english = stopwords.words('english')

#converting the list to a dictionary to make access time constant
stopwords_english = dict(Counter(stopwords_english))

"""
	This function builds a parent_dictionary.json file.
	This file is built by iterating over all the presidential debates.
	I have created a dictionary of dictionaries. For each debate, I have
	stored its corresponding word count in a dictionary format.
	I have used this word count to calculate tf_idf_dictionary in build_tf_idf_dict()
	function.
	This dictionary is stored in the form of :
	
	{
		document_1 : {
			word_1 : count_1,
			word_2 : count_2,
			.
			.
			.
			},

		document_2 : {
			word_1 : count_1,
			word_2 : count_2,
			.
			.
			.
			}
			.
			.
			.
	}
	format

	Since it is a dictionary of dictionaries, all access times are constant. i.e. O(1)
"""

def build_parent_dict() :

	corpusroot = './presidential_debates'

	stemmer = PorterStemmer()
	
	parent_dict = {}

	for filename in os.listdir(corpusroot):
		
		file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
		doc = file.read()
		file.close() 
		doc = doc.lower()

		doc = doc.encode(sys.stdout.encoding, errors='ignore') #extra

		#print(doc)

		tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
		#tokens = tokenizer.tokenize(doc)
		tokens = tokenizer.tokenize(str(doc)) #extra

		i = 0
		for token in tokens :
						
			if token not in stopwords_english :
				token = token.strip()
				token = stemmer.stem(token)
				tokens[i] = token
				#print(token)

			else :
				tokens[i] = None
				#print(token)

			i += 1

		temp_dictionary = dict(Counter(tokens))

		parent_dict[filename.split('.')[0]] = temp_dictionary
		
		#print(temp_dictionary)
		#print(tokens)
		#break

	import json
	with open('./parent_dictionary.json', 'w') as f:

		json.dump(parent_dict, f, indent=4)

"""
This function uses the parent_dict to build a document frequency for each word.
This is a simple dictionary which is stored in the form of :
	{
		word_1 : count_1,
		word_2 : count_2,
		.
		.
		.
	}
	format
"""

def build_idf_dict() :

	import json

	idf_dict = {}

	parent_dict = {}

	with open('./parent_dictionary.json', 'r') as f :

		try :
			parent_dict = json.load(f)
		
		except ValueError :
			parent_dict = {}

	for filename in parent_dict :

		temp_dict = parent_dict[filename]

		for word in temp_dict :

			if word in idf_dict :

				idf_dict[word] += 1

			else :

				idf_dict[word] = 1


	with open('./idf_dictionary.json', 'w') as f :

		json.dump(idf_dict, f, indent=4)


"""
This function builds a tf_idf_dictionary by fetching values from the 2 dictionaries
built previously.
The formula used here is : (1 + log10(parent_dict[term])) * log10(30/idf_dict[term])
"""
def build_tf_idf_dict() :

	tf_idf_dict = {}

	import json

	idf_dict = {}

	parent_dict = {}

	with open('./parent_dictionary.json', 'r') as f :

		try :
			parent_dict = json.load(f)
		
		except ValueError :
			parent_dict = {}

	with open('./idf_dictionary.json', 'r') as f :

		try :
			idf_dict = json.load(f)
		
		except ValueError :
			idf_dict = {}


	#main logic

	from math import log10

	for filename in parent_dict :

		current_dict = parent_dict[filename]

		temp_dict = {}

		for word in current_dict :

			temp_dict[word] = (1 + log10(current_dict[word])) * log10(30/idf_dict[word])

		tf_idf_dict[filename] = temp_dict

	#pretty_print(tf_idf_dict)

	tf_idf_dict = normalize(tf_idf_dict)

	#pretty_print(tf_idf_dict)

	with open('./tf_idf_dictionary.json', 'w') as f :

		json.dump(tf_idf_dict, f, indent=4)


"""
This function builds a query vector for the given query.
It also calculates and returns the most relevant document.
"""
def build_query_vector(query):

	query = query.lower()

	tokens = query.split(" ")

	tokens_dict = {}

	stemmer = PorterStemmer()

	from math import log10

	for token in tokens :
		
		if token not in stopwords_english :

			token = stemmer.stem(token)

			if token not in tokens_dict :
				tokens_dict[token] = 1

			else :
				tokens_dict[token] += 1

	for token in tokens_dict :

		tokens_dict[token] = 1 + log10(tokens_dict[token])

	tokens_dict = normalize(tokens_dict)

	#print(tokens_dict)

	tf_idf_dict = {}

	import json

	with open('./tf_idf_dictionary.json', 'r') as f :

		try :
			tf_idf_dict = json.load(f)
		
		except ValueError :
			tf_idf_dict = {}

	temp_dict = {}

	for token in tokens_dict :

		inner_temp_dict = {}

		for filename in tf_idf_dict :

			if token in tf_idf_dict[filename] :

				inner_temp_dict[filename] = tf_idf_dict[filename][token]

		temp_dict[token] = inner_temp_dict

	import operator
	for d in temp_dict :

		temp_dict[d] = sorted(temp_dict[d].items(), key=operator.itemgetter(1), reverse=True)

	#pretty_print(temp_dict)

	dict_to_return = {}

	for query_token in temp_dict :

		list = temp_dict[query_token]

		x = min(len(list), 10)

		for i in range(0, x) :

			tuple = list[i]
			
			if tuple[0] in dict_to_return :

				dict_to_return[tuple[0]] += tuple[1] * tokens_dict[query_token]

			else :

				dict_to_return[tuple[0]] = tuple[1] * tokens_dict[query_token]

	if len(dict_to_return) >= 1 :
		return (sorted(dict_to_return.items(), key=operator.itemgetter(1), reverse=True)[0])

	else :
		return (None, 0)

"""
This function normalizes the dictionaries
"""
def normalize(outer_dict) :

	from math import pow

	divide_by = 0

	for k in outer_dict :
		
		if type(outer_dict[k]) == dict :
			
			outer_dict[k] = normalize(outer_dict[k])
		
		else :
			
			for key in outer_dict :

				divide_by += pow(outer_dict[key], 2)

			divide_by = pow(divide_by, 0.5)

			#print (divide_by)

			for key in outer_dict :

				outer_dict[key] = float(outer_dict[key]) / float(divide_by)

		divide_by = 0

	return outer_dict

"""
############################## UTILITY FUNCTIONS ############################################
"""

def getidf(token) :

	import json
	from math import log10

	with open('./idf_dictionary.json', 'r') as f :

		try :
			idf_dict = json.load(f)
		
		except ValueError :
			idf_dict = {}

	try :
		return log10(30/idf_dict[token])

	except KeyError :
		return 0

def getweight(document, token) :

	import json

	with open('./tf_idf_dictionary.json', 'r') as f :

		try :
			tf_idf_dict = json.load(f)
		
		except ValueError :
			tf_idf_dict = {}

	try :
		return tf_idf_dict[document.split(".")[0]][token]

	except KeyError :
		return 0


def query(q) :

	return build_query_vector(q)

def pretty_print(object) :

	import pprint

	pp = pprint.PrettyPrinter(indent=1)

	pp.pprint(object)

"""
These functions only need to be called in the first run to build the dictionaries,
and their corresponding json files.

You do not need to call these functions for future runs.
All data is stored in json format in the memory, so you can just comment out these calls.
"""
build_parent_dict()

build_idf_dict()

build_tf_idf_dict()

print("(%s, %.12f)" % query("security conference ambassador"))

print("%.12f" % getweight("2012-10-16.txt","hispan"))

print("%.12f" % getidf("health"))

print("%.12f" % getidf("agenda"))