from random import *
import cv2


def randomInput():
    """This function generates uniform inputs to emulate the input given by the Machine Learning algorithm"""
    max = 1.0

    # Randomly distribute max probability
    value_1 = round(uniform(0.0, max), 8)
    max -= value_1
    value_2 = round(uniform(0.0, max), 8)
    max -= value_2
    value_3 = round(uniform(0.0, max), 8)
    max -= value_3
    value_4 = round(uniform(0.0, max), 8)
    max -= value_4
    value_5 = round(uniform(0.0, max), 8)
    max -= value_5
    value_6 = round(max, 8)
    probabilities = [value_1, value_2, value_3, value_4, value_5, value_6]
    # FOR TESTING PURPOSES
    #probabilities = [0.17666491, 0.72653016, 0.01022988, 0.01623115, 0.06900918, 0.00133472]

    #Randomly distribute all probabilities among 6 possible emotions
    random_index = randint(0, len(probabilities) - 1)
    ANGER = probabilities.pop(random_index)  # ANGER

    random_index = randint(0, len(probabilities) - 1)
    DISGUST = probabilities.pop(random_index)  # DISGUST

    random_index = randint(0, len(probabilities) - 1)
    FEAR = probabilities.pop(random_index)  # FEAR

    random_index = randint(0, len(probabilities) - 1)
    HAPPY = probabilities.pop(random_index)  # HAPPY

    random_index = randint(0, len(probabilities) - 1)
    SADNESS = probabilities.pop(random_index)  # SADNESS

    SURPRISE = probabilities[0]  # SURPRISE

    input = [ANGER, DISGUST, FEAR, HAPPY, SADNESS, SURPRISE]

    #Print input to screen
    print("RANDOMIZED INPUT\n" + str(input) + "\n" + str(round(sum(input), 8)) + "\n")

    return input


def getEmotion(index):
    """Maps the given index on the input array to the corresponding emotion, returns label as string"""
    if (index == 0):
        return "ANGER"
    elif (index == 1):
        return "DISGUST"
    elif (index == 2):
        return "FEAR"
    elif (index == 3):
        return "HAPPY"
    elif (index == 4):
        return "SADNESS"
    elif (index == 5):
        return "SURPRISE"


def playVideo(video_index):
    if (video_index == 0):  # Input -> ANGER -> Output -> FEAR
        cap = cv2.VideoCapture('videos/01.flv')
    elif (video_index == 1):  # Input -> DISGUST -> Output -> DISGUST
        cap = cv2.VideoCapture('videos/02.flv')
    elif (video_index == 2):  # Input -> FEAR -> Output -> SURPRISE
        cap = cv2.VideoCapture('videos/01.flv')
    elif (video_index == 3):  # Input -> HAPPY -> Output -> FEAR
        cap = cv2.VideoCapture('videos/02.flv')
    elif (video_index == 4):  # Input -> SADNESS -> Output -> SADNESS
        cap = cv2.VideoCapture('videos/01.flv')
    elif (video_index == 5):  # Input -> SURPRISE -> Output -> SURPRISE
        cap = cv2.VideoCapture('videos/02.flv')

    while (cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def emotionSynthesis(input):
    highest = max(input)  # find highest probability

    for index, probability in enumerate(input):
        print(getEmotion(index) + " => " + str(probability))  # print entire input

        if (probability == highest):
            highest_index = index  # get index

    print("\nHighest Value => " + str(highest))
    print("Most Likely Emotion => " + getEmotion(highest_index))

    # Play corresponding emotion video
    if (highest_index == 0):
        playVideo(0)
    elif (highest_index == 1):
        playVideo(1)
    elif (highest_index == 2):
        playVideo(2)
    elif (highest_index == 3):
        playVideo(3)
    elif (highest_index == 4):
        playVideo(4)
    elif (highest_index == 5):
        playVideo(5)


def main():
    input = randomInput()
    emotionSynthesis(input)


if __name__ == "__main__":
    main()
