from searcher.classifier import Classifier
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dst", type = str, help = "The src directory for training.", required = False)
    parser.add_argument("-t", "--train_dir", type = str, help = "Training directory where te model will be saved", required = True)

    args = parser.parse_args()


    cls = Classifier(data_dir_path = args.src_dst)
    cls.fit(epochs = 2, save_dir_path = args.train_dir)






