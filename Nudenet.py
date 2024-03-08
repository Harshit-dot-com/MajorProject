from nudenet import NudeDetector

def NudityDetector(pathToImage):
    nudityExists = False
    nude_detector = NudeDetector()
    preds = nude_detector.detect(pathToImage)
    for pred in preds:
        print(pred)
        if(pred["class"] == "FEMALE_BREAST_EXPOSED"):
            nudityExists = True
            print("female breasts detected")
        
        if(pred["class"] == "BUTTOCKS_EXPOSED"):
            nudityExists = True
            print("buttocks detected")

        if(pred["class"] == "FEMALE_GENITALIA_EXPOSED"):
            nudityExists = True
            print("Female genetailia detected")

        if(pred["class"] == "ANUS_EXPOSED"):
            nudityExists = True
            print("Anus detected")

        if(pred["class"] == "MALE_GENITALIA_EXPOSED"):
            nudityExists = True
            print("Male Genetalia detected")

    return nudityExists