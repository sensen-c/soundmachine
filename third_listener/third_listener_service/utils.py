mood_dict = {
    0: 'angry',
    1: 'happy',
    2: 'excited',
    3: 'sad',
    4: 'frustrated',
    5: 'fearful',
    6: 'surprised',
    7: 'neutral',
    8: 'other'
}
def convert_prediction_to_str(prediction):
    return mood_dict.get(prediction)