import math

def calculate_naive_probability(model_n_prior, sentence, value):
    model, model_prior_probability = model_n_prior
    
    log_probability_sum = model_prior_probability[value]

    for word in sentence.split():
        if word not in model[value].keys():
            continue
        log_probability_sum += model[value][word]

    return log_probability_sum

def train_the_model(data, labels):
    max_label = max(labels)+1
    vocab = list(set(data.keys()))
    model = [dict() for _ in range(max_label)]
    model_prior_probability = [None] * max_label
    for value in range(max_label):
        model_prior_probability[value] = math.log ( labels.count(value) / len(labels) )

        total_occurrences = 0
        for i in range(len(data[list(data.keys())[0]])):
            if labels[i] == value:
                total_occurrences += sum([data[key][i] for key in data.keys()])

        for word in vocab:
            word_occurrences = sum([item[0] for item in zip(data[word], labels) if item[1] == value])
            model[value][word] = math.log((word_occurrences + 1) / (total_occurrences + len(data.keys())))
    return (model, model_prior_probability)

training_data = ["just plain boring",
                 "entirely predictable and lacks energy",
                 "no surprises and very few laughs",
                 "very powerful", "the most fun film of the summer"]
training_labels = [0, 0, 0, 1, 1]

def process_raw_data(raw_data):
    merged_data = []
    for item in raw_data:
        merged_data += item.split()
    data_set = {}
    vocabulary = list(set(merged_data))
    for key in vocabulary:
        data_set[key] = [item.split().count(key) for item in raw_data]
    return data_set

data_set = process_raw_data(training_data)
model_n_prior = train_the_model(data_set, training_labels)


sentence_to_classify = "predictable with no fun"

positive_probability = calculate_naive_probability(model_n_prior, sentence_to_classify, 1)
negative_probability = calculate_naive_probability(model_n_prior, sentence_to_classify, 0)
print(math.exp(positive_probability), math.exp(negative_probability))
if math.exp(positive_probability) > math.exp(negative_probability):
    print("positive")
else:
    print("negative")
