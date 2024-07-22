import cv2
import numpy as np
import googletrans
import streamlit as st

translator = googletrans.Translator()


def fourPointsTransform(frame, vertices):
    """Extracts and transforms roi of frame defined by vertices into a rectangle."""
    # Print vertices of each bounding box.
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    # Apply perspective transform
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)

    return result


# Set title.
st.title("OpenCV Image to Text Translator")

# Inverse mapping of Language for easy readability
langDict = {val: key for key, val in googletrans.LANGUAGES.items()}

# Add an extra language
detectLangs = langDict.copy()
detectLangs["Detect"] = "auto"  # `auto` is a library keyword
detectSelection = st.sidebar.selectbox(
    "Translate from :", detectLangs, index=list(detectLangs.values()).index("auto"))
src_lang = detectLangs[detectSelection]

languageSelection = st.sidebar.selectbox(
    "Translate into :", langDict, index=list(langDict.values()).index("en"))
dest_lang = langDict[languageSelection]

# Upload image.
uploaded_file = st.sidebar.file_uploader("Choose a text image", type="jpg")


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    inp_col, op_col = st.columns(2)

    # Define list to store the vocabulary in
    vocabulary = []

    # Open file to import the vocabulary.
    with open("resources/alphabet_94.txt") as f:
        # Read the file line by line, and append each into the vocabulary list.
        for line in f:
            vocabulary.append(line.strip())
        f.close()

    # DB model for text-detection based on resnet50.
    textDetector = cv2.dnn_TextDetectionModel_DB("resources/DB_TD500_resnet50.onnx")

    inputSize = (640, 640)

    # Set threshold for Binary Map creation and polygon detection
    binThresh = 0.3
    polyThresh = 0.5

    mean = (122.67891434, 116.66876762, 104.00698793)

    textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
    textDetector.setInputParams(1.0/255, inputSize, mean, True)

    # CRNN model for text-recognition.
    textRecognizer = cv2.dnn_TextRecognitionModel("resources/crnn_cs.onnx")
    textRecognizer.setDecodeType("CTC-greedy")
    textRecognizer.setVocabulary(vocabulary)
    textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)

    # Create a blank matrix to be used to display the output image.
    outputCanvas = np.full(image.shape[:3], 255, dtype=np.uint8)

    # Use the DB text detector initialised previously to detect the presence of text in the image.
    boxes, confs = textDetector.detect(image)
    imageDisplay = image.copy()
    cv2.polylines(imageDisplay, boxes, True, (255, 0, 255), 3)

    with inp_col:
        st.header("Detected Text")
        st.image(imageDisplay[:, :, ::-1])

    # Iterate through the bounding boxes detected by the text detector model.
    for box in boxes:

        # Apply transformation on the detected text bounding box.
        croppedRoi = fourPointsTransform(image, box)

        # Recognise the text using the crnn model.
        recognizedText = textRecognizer.recognize(croppedRoi)
        translation = translator.translate(recognizedText, dest_lang, src_lang)

        st.write("Recognized Text[{}]: {} -> Translated Text[{}]: {}".format(
            googletrans.LANGUAGES[translation.src],
            recognizedText,
            googletrans.LANGUAGES[dest_lang],
            translation.text))

        # Get scaled values.
        boxHeight = int((abs((box[0, 1] - box[1, 1]))))

        # Get scale of the font.
        fontScale = cv2.getFontScaleFromHeight(
            cv2.FONT_HERSHEY_SIMPLEX, boxHeight-5, 1)

        # Write the recognised text on the output image.
        placement = (int(box[0, 0]), int(box[0, 1]))
        cv2.putText(outputCanvas, translation.text, placement,
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 3, 3)

    with op_col:
        st.header('Translated Text')
        st.image(outputCanvas[:, :, ::-1])