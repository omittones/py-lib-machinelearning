import cntk
import numpy as np
#import seaborn as sns
from cntk.learners import sgd
from cntk.logging import ProgressPrinter
from cntk.layers import Dense, Sequential

def xor(x, y):
    return (x > 0.5) != (y > 0.5)

def generate_xor_data():
    space = np.linspace(0, 1, 11, True, dtype = np.float32)
    data = [[x, y, xor(x,y)] for x in space for y in space]
    return np.asarray(data)

def generate_xor_set(sample_size):
    def tosmax(bool):
        return [0, 1] if bool else [1, 0]
    X = np.random.random(size=(sample_size, 2))
    X = X.astype(np.float32)
    Y = np.asarray([tosmax(xor(i[0], i[1])) for i in X], dtype=np.float32)
    return X, Y

def generate_random_data(sample_size):
    feature_dim = 2
    num_classes = 2

    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
    X = X.astype(np.float32)
    # converting class 0 into the vector "1 0 0",
    # class 1 into vector "0 1 0", ...
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y

def ffnet():
    
    device = cntk.device.gpu(0)
    cntk.device.try_set_default_device(device)
    np.random.seed(98052)
    
    inputs = 2
    outputs = 2
    layers = 2
    hidden_dimension = 50
    gen_data = generate_xor_set

    # input variables denoting the features and label data
    var_feature = cntk.input_variable((inputs), np.float32)
    var_label = cntk.input_variable((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential([
                    Dense(hidden_dimension, activation=cntk.leaky_relu),
                    Dense(outputs)])
    z = my_model(var_feature)

    ce = cntk.cross_entropy_with_softmax(z, var_label)
    pe = cntk.classification_error(z, var_label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = cntk.learning_parameter_schedule(0.125)
    progress_printer = ProgressPrinter(freq=10)
    trainer = cntk.Trainer(z, (ce, pe), [sgd(z.parameters, lr=lr_per_minibatch)], [progress_printer])

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_minibatches_to_train = 100

    minibatches = [gen_data(minibatch_size) for i in range(num_minibatches_to_train)]

    aggregate_loss = 0.0

    #sns.set(style="whitegrid")
    #ax = sns.stripplot(x=[]tips["total_bill"])
    data = generate_xor_data()

    for epoch in range(100):
        for (features, labels) in minibatches:
            trainer.train_minibatch({var_feature : features, var_label : labels}, device=device)
            sample_count = trainer.previous_minibatch_sample_count
            aggregate_loss += trainer.previous_minibatch_loss_average * sample_count

    last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen

    test_features, test_labels = gen_data(minibatch_size)
    avg_error = trainer.test_minibatch({var_feature : test_features, var_label : test_labels})
    print('    error rate on an unseen minibatch: {}'.format(avg_error))
    return last_avg_error, avg_error

#import clr
#from System import String
#from System.Collections import ArrayList
#clr.AddReference("System.Windows.Forms")
#from System.Windows.Forms import Form
#clr.AddReference("D:/Code/cs-lib-bitcoin-predictions/omittones/trader/bin/Debug/trader.dll")
#from Core import SimulatedTrader
#obj = SimulatedTrader(None, None)
#pass