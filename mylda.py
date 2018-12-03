import torch as tc
import numpy as np


def lda(num_topics, iterations, alpha, beta):
    documents = [] # TODO: figure this out
    vocabulary = [] # TODO: figure this out
    print("Starting gibbs sampler")
    num_documents = len(documents) # TODO: Arvind
    vocabulary_size = 10 # # TODO: Arvind

    # Arrays that will count through the lda process
    document_counts = np.zeros((num_documents, num_topics), dtype=np.int)
    topic_word_counts = np.zeros((num_topics, vocabulary_size), dtype=np.int)
    word_topic_assigments = []
    topic_counts = np.zeros((num_topics), dtype=np.int)

    for d_idx, document in enumerate(documents):
        w_assigns = []
        for word in document.words:
            if word in vocabulary:
                # Select random starting topic assignment for word.
                w_idx = vocabulary.index(word)
                starting_topic_index = np.random.randint(num_topics) # randomly assign topic to every word
                w_assigns.append(starting_topic_index)
                # Set current topic assignment, increment doc-topic and word-topic counters.
                document_counts[d_idx, starting_topic_index] += 1
                topic_word_counts[starting_topic_index, w_idx] += 1
                topic_counts[starting_topic_index] += 1
        word_topic_assigments.append(np.array(w_assigns))

    # Run the sampler.
    for iteration in range(iterations):
        print ("Iteration #" + str(iteration + 1) + "...")
        for d_index, document in enumerate(documents):
            for w, word in enumerate(document.words):
                if word in vocabulary:
                    w_idx = vocabulary.index(word)
                    # Get the topic that the word is currently assigned to.
                    current_topic_index = word_topic_assigments[d_index][w]
                    # Decrement counts.
                    document_counts[d_index, current_topic_index] -= 1
                    topic_word_counts[current_topic_index, w_idx] -= 1
                    topic_counts[current_topic_index] -= 1
                    # Get new topic.
                    topic_distribution = (topic_word_counts[:, w_idx] + beta) *
                        (document_counts[d_index] + alpha)
                        (topic_counts + beta) # changed by hitalex
                    #new_topic_index = np.random.multinomial(1, np.random.dirichlet(topic_distribution)).argmax()
                    # choose a new topic index according to topic distribution
                    # new_topic_index = choose(range(number_of_topics), topic_distribution)
                    # # Reassign and notch up counts.
                    # self.current_word_topic_assignments[d_index][w] = new_topic_index
                    # self.document_topic_counts[d_index, new_topic_index] += 1
                    # self.topic_word_counts[new_topic_index, w_index] += 1
                    # self.topic_counts[new_topic_index] += 1
