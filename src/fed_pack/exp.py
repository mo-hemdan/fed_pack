
from trainer import ClientTrainer
import utils

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
monitor_ip = "188.185.11.183"
# monitor_ip = "188.185.120.31"
server_ip = "188.185.11.183"
client_id = "DefaultHOSTNAME"

def main() -> None:
    # Parse command line argument `partition`
    global client_id
    print(os.environ['client_ip'])
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=str, required=True) #choices=range(0, 10), )
    parser.add_argument("--client_id", type=str, required=True) #choices=range(0, 10), )
    parser.add_argument("--filename", type=str, required=True) #choices=range(0, 10), )
    args = parser.parse_args()
    client_id = args.client_id
    print("Client ID: ", client_id)
    print("Partition: ", int(args.partition))

    # Load and compile Keras model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    # path = "/data/data_batch_3"
    path = "/data/" + args.filename
    x_train, x_test, y_train, y_test = load_cifar(path)
    # (x_train, y_train), (x_test, y_test) = load_partition(int(args.partition))

    # Start Flower client
    def train_func(model, x, y, batch_size, epochs):
        history = model.fit(
            x,
            y,
            batch_size,
            epochs,
            validation_split=0.1,
        )
        return history
    
    client_tr = ClientTrainer()
    client_tr.set_data(x_train, x_test, y_train, y_test)
    client_tr.set_model(model)
    client_tr.train(server_ip, train_func)


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (
        x_train[idx * 5000 : (idx + 1) * 5000],
        y_train[idx * 5000 : (idx + 1) * 5000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )

def load_cifar(path):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    dataset = unpickle(path) #
    import numpy as np
    data = dataset[b'data']
    labels = np.array(dataset[b'labels']).reshape((-1, 1))
    data_reshaped = data.reshape((-1, 3, 32, 32))
    data_transposed = np.transpose(data_reshaped, axes=[0, 2, 3, 1])
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_transposed, labels, random_state=0, test_size= 0.2)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()