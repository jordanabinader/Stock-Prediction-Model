import matplotlib.pyplot as plt

# Function to plot testing predictions and observations
def plot_testing_predictions(dates_test, test_predictions, y_test):
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Testing Predictions', 'Testing Observations'])
    plt.show()

# Function to plot validation predictions and observations
def plot_validation_predictions(dates_val, val_predictions, y_val):
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.legend(['Validation Predictions', 'Validation Observations'])
    plt.show()

# Function to plot training, validation, and testing predictions and observations
def plot_training_validation_testing_predictions(dates_train, train_predictions,
                                                 dates_val, val_predictions,
                                                 dates_test, test_predictions,
                                                 y_train, y_val, y_test):
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.legend(['Training Predictions', 'Training Observations',
                'Validation Predictions', 'Validation Observations',
                'Testing Predictions', 'Testing Observations'])
    plt.show()

# Function to plot training, validation, testing, and recursive predictions and observations
def plot_all_predictions(dates_train, train_predictions,
                         dates_val, val_predictions,
                         dates_test, test_predictions,
                         recursive_dates, recursive_predictions,
                         y_train, y_val, y_test):
    plt.plot(dates_train, train_predictions)
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, val_predictions)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, test_predictions)
    plt.plot(dates_test, y_test)
    plt.plot(recursive_dates, recursive_predictions)
    plt.legend(['Training Predictions', 'Training Observations',
                'Validation Predictions', 'Validation Observations',
                'Testing Predictions', 'Testing Observations',
                'Recursive Predictions'])
    plt.show()
