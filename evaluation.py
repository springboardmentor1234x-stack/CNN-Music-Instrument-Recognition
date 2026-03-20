from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(y_true, y_pred):

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    plt.figure()
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")