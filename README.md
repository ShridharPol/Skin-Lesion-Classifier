# Skin Lesion Classification using MobileNetV2

This project is a **lightweight prototype** for a skin lesion classification system built using **MobileNetV2** and deployed via a **Streamlit web app**. The main goal is to explore the feasibility of deploying deep learning–based diagnostic tools in **low-resource settings**, where access to expert dermatological care is limited.

---

## Motivation

Early detection of skin cancers such as **melanoma** is critical. However, access to dermatologists is often lacking in underserved areas. This project aims to:

- Build a **lightweight deep learning model** that can run on mobile or low-end devices.
- Evaluate whether pre-trained models like MobileNetV2 are viable for this task.
- Deploy a **minimal web-based app** for skin lesion prediction.
- Lay the groundwork for future improvements using **GANs**, **segmentation**, or **clinical data**.

---

## Project Structure

Skin_Lesion_Classifier/
├── README.md
├── requirements.txt
├── Skin_Lesion_Classification_MobileNetv2.ipynb
├── App/
│ ├── app.py
│ └── best_model_final_MobNet.keras

## Dataset

This model was trained on the **HAM10000** dataset — a publicly available collection of dermatoscopic images for 7 skin lesion classes.

- **Note**: Dataset preprocessing included class balancing via **data augmentation** only.
- No semantic-level diversity was added (e.g., via GANs).

---

## How to Run the App

1. Clone the repo:
    ```bash
    git clone https://github.com/your-username/Skin_Lesion_Classifier.git
    cd Skin_Lesion_Classifier/App
    ```

2. Install dependencies:
    ```bash
    pip install -r ../requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

4. Open `http://localhost:8501` in your browser.

---

## Limitations

- The model shows **overfitting**: training accuracy approaches 100%, while validation accuracy is lower.
- **Data augmentation** helped mitigate imbalance but doesn’t create new information.
- **Precision, recall, and F1-scores** are suboptimal for minority classes.
- No **literature survey** or **clinical validation** has been conducted.
- No **segmentation** or lesion localization techniques used.

---

## Future Scope

- Apply **GANs** to generate synthetic images for underrepresented classes.
- Use **focal loss** or smarter sampling techniques to emphasize hard examples.
- Explore **quantization-aware training** to deploy on edge devices.
- Perform **literature review** and consult domain experts for feedback.
- Add **segmentation** to focus the model on lesion areas only.

---

## Acknowledgements

- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- Built with TensorFlow, Keras, and Streamlit

---

## Author

**Shridhar Pol**  
MS in Electrical and Computer Engineering  
Northeastern University  
Passionate about using AI for social good
