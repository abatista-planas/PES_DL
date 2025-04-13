from src.pes_1D.predictor import Predictor

if __name__ == "__main__":
    predictor = Predictor("./src/")
    y = predictor.predict_one([1.2, 0.8, 2.4, 1.2])
    print(y)

    y_batch = predictor.predict_batch([[1.2, 0.8, 2.4, 1.2], [2.1, 1.4, 1.0, 3.2]])
    print(y_batch)
