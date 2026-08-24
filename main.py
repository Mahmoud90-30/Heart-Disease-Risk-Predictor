import sys
from pathlib import Path

import joblib
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


MODEL_PATH = Path("model.pkl")


class HeartGuard(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "HeartGuard | Heart Disease Predictor"
        )

        self.setMinimumSize(1100, 750)

        self.model = None

        self.load_model()
        self.build_ui()

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(self):

        if MODEL_PATH.exists():

            try:
                self.model = joblib.load(MODEL_PATH)

            except Exception as e:
                self.model = None
                print("Model loading error:", e)

    # =====================================================
    # Build UI
    # =====================================================

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        root.setContentsMargins(
            30,
            28,
            30,
            24
        )

        root.setSpacing(20)

        # =================================================
        # HEADER
        # =================================================

        header = QFrame()

        header.setObjectName("header")

        header_layout = QHBoxLayout(header)

        header_layout.setContentsMargins(
            28,
            20,
            28,
            20
        )

        # Heart icon

        heart = QLabel("♥")

        heart.setObjectName(
            "heartIcon"
        )

        heart.setAlignment(
            Qt.AlignCenter
        )

        header_layout.addWidget(
            heart
        )

        # Header text

        header_text = QVBoxLayout()

        title = QLabel(
            "HeartGuard"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "Heart Disease Risk Prediction"
            " • Machine Learning"
        )

        subtitle.setObjectName(
            "subtitle"
        )

        header_text.addWidget(
            title
        )

        header_text.addWidget(
            subtitle
        )

        header_layout.addLayout(
            header_text
        )

        header_layout.addStretch()

        # AI badge

        ai_badge = QLabel(
            "AI POWERED"
        )

        ai_badge.setObjectName(
            "aiBadge"
        )

        ai_badge.setAlignment(
            Qt.AlignCenter
        )

        header_layout.addWidget(
            ai_badge
        )

        root.addWidget(
            header
        )

        # =================================================
        # MAIN CONTENT
        # =================================================

        body = QHBoxLayout()

        body.setSpacing(20)

        root.addLayout(
            body,
            1
        )

        # =================================================
        # INPUT CARD
        # =================================================

        input_card = QFrame()

        input_card.setObjectName(
            "card"
        )

        input_layout = QVBoxLayout(
            input_card
        )

        input_layout.setContentsMargins(
            26,
            24,
            26,
            26
        )

        input_layout.setSpacing(16)

        input_title = QLabel(
            "Patient Information"
        )

        input_title.setObjectName(
            "cardTitle"
        )

        input_layout.addWidget(
            input_title
        )

        input_desc = QLabel(
            "Enter the patient's clinical "
            "measurements and information."
        )

        input_desc.setObjectName(
            "muted"
        )

        input_layout.addWidget(
            input_desc
        )

        # =================================================
        # INPUT GRID
        # =================================================

        grid = QGridLayout()

        grid.setHorizontalSpacing(
            18
        )

        grid.setVerticalSpacing(
            14
        )

        # Age

        self.age = QSpinBox()

        self.age.setRange(
            1,
            120
        )

        self.age.setValue(
            50
        )

        # Sex

        self.sex = self.combo(
            [
                "M",
                "F"
            ]
        )

        # Chest Pain

        self.chest_pain = self.combo(
            [
                "ATA",
                "NAP",
                "ASY",
                "TA"
            ]
        )

        # Resting BP

        self.resting_bp = QSpinBox()

        self.resting_bp.setRange(
            0,
            300
        )

        self.resting_bp.setValue(
            120
        )

        # Cholesterol

        self.cholesterol = QSpinBox()

        self.cholesterol.setRange(
            0,
            1000
        )

        self.cholesterol.setValue(
            200
        )

        # Fasting BS

        self.fasting_bs = self.combo(
            [
                "0",
                "1"
            ]
        )

        # Resting ECG

        self.resting_ecg = self.combo(
            [
                "Normal",
                "ST",
                "LVH"
            ]
        )

        # Max HR

        self.max_hr = QSpinBox()

        self.max_hr.setRange(
            0,
            300
        )

        self.max_hr.setValue(
            150
        )

        # Exercise Angina

        self.exercise_angina = self.combo(
            [
                "N",
                "Y"
            ]
        )

        # Oldpeak

        self.oldpeak = QDoubleSpinBox()

        self.oldpeak.setRange(
            -10,
            20
        )

        self.oldpeak.setDecimals(
            1
        )

        self.oldpeak.setSingleStep(
            0.1
        )

        # ST Slope

        self.st_slope = self.combo(
            [
                "Up",
                "Flat",
                "Down"
            ]
        )

        fields = [

            (
                "Age",
                self.age
            ),

            (
                "Sex",
                self.sex
            ),

            (
                "Chest Pain Type",
                self.chest_pain
            ),

            (
                "Resting Blood Pressure",
                self.resting_bp
            ),

            (
                "Cholesterol",
                self.cholesterol
            ),

            (
                "Fasting Blood Sugar",
                self.fasting_bs
            ),

            (
                "Resting ECG",
                self.resting_ecg
            ),

            (
                "Maximum Heart Rate",
                self.max_hr
            ),

            (
                "Exercise Angina",
                self.exercise_angina
            ),

            (
                "Oldpeak",
                self.oldpeak
            ),

            (
                "ST Slope",
                self.st_slope
            )

        ]

        for i, (label, widget) in enumerate(fields):

            row = i // 2

            col = (i % 2) * 2

            label_widget = QLabel(
                label
            )

            label_widget.setObjectName(
                "fieldLabel"
            )

            grid.addWidget(
                label_widget,
                row,
                col
            )

            grid.addWidget(
                widget,
                row,
                col + 1
            )

        input_layout.addLayout(
            grid
        )

        # =================================================
        # PREDICT BUTTON
        # =================================================

        self.predict_btn = QPushButton(
            "Predict Heart Disease Risk"
        )

        self.predict_btn.setObjectName(
            "predictButton"
        )

        self.predict_btn.clicked.connect(
            self.predict
        )

        input_layout.addWidget(
            self.predict_btn
        )

        body.addWidget(
            input_card,
            3
        )

        # =================================================
        # RESULT CARD
        # =================================================

        result_card = QFrame()

        result_card.setObjectName(
            "card"
        )

        result_layout = QVBoxLayout(
            result_card
        )

        result_layout.setContentsMargins(
            28,
            26,
            28,
            26
        )

        result_title = QLabel(
            "Prediction"
        )

        result_title.setObjectName(
            "cardTitle"
        )

        result_layout.addWidget(
            result_title
        )

        result_desc = QLabel(
            "AI-powered evaluation of the "
            "patient profile."
        )

        result_desc.setObjectName(
            "muted"
        )

        result_layout.addWidget(
            result_desc
        )

        # =================================================
        # Result Area
        # =================================================

        self.result_area = QFrame()

        self.result_area.setObjectName(
            "resultArea"
        )

        result_area_layout = QVBoxLayout(
            self.result_area
        )

        result_area_layout.setContentsMargins(
            20,
            30,
            20,
            30
        )

        result_area_layout.setAlignment(
            Qt.AlignCenter
        )

        self.icon = QLabel(
            "♥"
        )

        self.icon.setObjectName(
            "resultIcon"
        )

        self.icon.setAlignment(
            Qt.AlignCenter
        )

        result_area_layout.addWidget(
            self.icon
        )

        self.result = QLabel(
            "Ready for Prediction"
        )

        self.result.setObjectName(
            "resultTitle"
        )

        self.result.setAlignment(
            Qt.AlignCenter
        )

        result_area_layout.addWidget(
            self.result
        )

        self.result_text = QLabel(
            "Enter patient information "
            "and run the prediction."
        )

        self.result_text.setObjectName(
            "resultText"
        )

        self.result_text.setWordWrap(
            True
        )

        self.result_text.setAlignment(
            Qt.AlignCenter
        )

        result_area_layout.addWidget(
            self.result_text
        )

        result_layout.addWidget(
            self.result_area
        )

        # =================================================
        # Probability
        # =================================================

        probability_title = QLabel(
            "Model Probability"
        )

        probability_title.setObjectName(
            "smallTitle"
        )

        probability_title.setAlignment(
            Qt.AlignCenter
        )

        result_layout.addWidget(
            probability_title
        )

        self.probability = QLabel(
            "—"
        )

        self.probability.setObjectName(
            "probability"
        )

        self.probability.setAlignment(
            Qt.AlignCenter
        )

        result_layout.addWidget(
            self.probability
        )

        # =================================================
        # Model Status
        # =================================================

        result_layout.addStretch()

        self.status = QLabel(
            "● Model Loaded"
            if self.model is not None
            else "● Model Not Found"
        )

        self.status.setObjectName(
            "statusLoaded"
            if self.model is not None
            else "statusError"
        )

        self.status.setAlignment(
            Qt.AlignCenter
        )

        result_layout.addWidget(
            self.status
        )

        body.addWidget(
            result_card,
            2
        )

        # =================================================
        # DISCLAIMER
        # =================================================

        disclaimer = QLabel(
            "Educational project only. "
            "This prediction is not a medical diagnosis "
            "and should not replace professional medical advice."
        )

        disclaimer.setObjectName(
            "disclaimer"
        )

        disclaimer.setWordWrap(
            True
        )

        root.addWidget(
            disclaimer
        )

        # Apply styling

        self.apply_style()

    # =====================================================
    # ComboBox Helper
    # =====================================================

    def combo(
        self,
        values
    ):

        box = QComboBox()

        box.addItems(
            values
        )

        return box

    # =====================================================
    # Prediction
    # =====================================================

    def predict(self):

        if self.model is None:

            QMessageBox.warning(
                self,
                "Model Not Found",
                "Please place your trained model "
                "in the application folder as:\n\n"
                "model.pkl"
            )

            return

        data = pd.DataFrame(
            [
                {

                    "Age":
                        self.age.value(),

                    "Sex":
                        self.sex.currentText(),

                    "ChestPainType":
                        self.chest_pain.currentText(),

                    "RestingBP":
                        self.resting_bp.value(),

                    "Cholesterol":
                        self.cholesterol.value(),

                    "FastingBS":
                        int(
                            self.fasting_bs.currentText()
                        ),

                    "RestingECG":
                        self.resting_ecg.currentText(),

                    "MaxHR":
                        self.max_hr.value(),

                    "ExerciseAngina":
                        self.exercise_angina.currentText(),

                    "Oldpeak":
                        self.oldpeak.value(),

                    "ST_Slope":
                        self.st_slope.currentText()

                }
            ]
        )

        try:

            prediction = int(
                self.model.predict(
                    data
                )[0]
            )

            probability = None

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probability = float(
                    self.model.predict_proba(
                        data
                    )[0][1]
                )

            # =================================================
            # HIGH RISK
            # =================================================

            if prediction == 1:

                self.result_area.setObjectName(
                    "highRiskArea"
                )

                self.result_area.style().unpolish(
                    self.result_area
                )

                self.result_area.style().polish(
                    self.result_area
                )

                self.icon.setText(
                    "!"
                )

                self.icon.setObjectName(
                    "highRiskIcon"
                )

                self.result.setText(
                    "Higher Risk Detected"
                )

                self.result.setStyleSheet(
                    "color: #C45A5A;"
                )

                self.result_text.setText(
                    "The model predicts a positive "
                    "heart-disease class for this profile."
                )

            # =================================================
            # LOW RISK
            # =================================================

            else:

                self.result_area.setObjectName(
                    "lowRiskArea"
                )

                self.result_area.style().unpolish(
                    self.result_area
                )

                self.result_area.style().polish(
                    self.result_area
                )

                self.icon.setText(
                    "✓"
                )

                self.icon.setObjectName(
                    "lowRiskIcon"
                )

                self.result.setText(
                    "Lower Risk Detected"
                )

                self.result.setStyleSheet(
                    "color: #4E9B78;"
                )

                self.result_text.setText(
                    "The model predicts a negative "
                    "heart-disease class for this profile."
                )

            # =================================================
            # Probability
            # =================================================

            if probability is not None:

                self.probability.setText(
                    f"{probability:.1%}"
                )

            else:

                self.probability.setText(
                    "Available after prediction"
                )

        except Exception as e:

            QMessageBox.critical(
                self,
                "Prediction Error",
                "The model could not process "
                "the submitted data.\n\n"
                "Make sure model.pkl contains "
                "the complete preprocessing + "
                "model pipeline.\n\n"
                f"Details:\n{e}"
            )

    # =====================================================
    # Modern Medical Theme
    # =====================================================

    def apply_style(self):

        self.setStyleSheet("""

        /* ==========================================
           GLOBAL
        ========================================== */

        QMainWindow {

            background-color: #F5F8FA;

        }

        QWidget {

            font-family: "Segoe UI";

            color: #334155;

        }


        /* ==========================================
           HEADER
        ========================================== */

        #header {

            background-color: #E8F4F5;

            border: 1px solid #D6E9EB;

            border-radius: 18px;

        }

        #heartIcon {

            background-color: #D8EEF0;

            color: #4A9AA0;

            border-radius: 18px;

            min-width: 50px;

            max-width: 50px;

            min-height: 50px;

            max-height: 50px;

            font-size: 28px;

            font-weight: bold;

        }

        #title {

            color: #23404A;

            font-size: 29px;

            font-weight: 700;

        }

        #subtitle {

            color: #6B8790;

            font-size: 12px;

        }

        #aiBadge {

            background-color: #DFF2ED;

            color: #43836A;

            border: 1px solid #C9E8DD;

            border-radius: 10px;

            padding: 7px 12px;

            font-size: 10px;

            font-weight: 700;

        }


        /* ==========================================
           CARDS
        ========================================== */

        #card {

            background-color: #FFFFFF;

            border: 1px solid #E3EAED;

            border-radius: 18px;

        }

        #cardTitle {

            color: #263E47;

            font-size: 19px;

            font-weight: 700;

        }

        #muted {

            color: #78909A;

            font-size: 12px;

        }


        /* ==========================================
           FIELD LABELS
        ========================================== */

        #fieldLabel {

            color: #536A72;

            font-size: 11px;

            font-weight: 600;

        }


        /* ==========================================
           INPUTS
        ========================================== */

        QSpinBox,
        QDoubleSpinBox,
        QComboBox {

            background-color: #F8FAFB;

            border: 1px solid #D9E3E6;

            border-radius: 9px;

            padding: 8px 10px;

            min-height: 22px;

            color: #334E57;

            font-size: 12px;

        }

        QSpinBox:hover,
        QDoubleSpinBox:hover,
        QComboBox:hover {

            border: 1px solid #9BC8CC;

        }

        QSpinBox:focus,
        QDoubleSpinBox:focus,
        QComboBox:focus {

            background-color: #FFFFFF;

            border: 1px solid #6BAEB4;

        }

        QComboBox QAbstractItemView {

            background-color: white;

            border: 1px solid #D9E3E6;

            selection-background-color: #E5F2F3;

            selection-color: #29464F;

        }


        /* ==========================================
           BUTTON
        ========================================== */

        #predictButton {

            background-color: #5A9FA5;

            color: white;

            border: none;

            border-radius: 10px;

            min-height: 46px;

            font-size: 13px;

            font-weight: 700;

            margin-top: 7px;

        }

        #predictButton:hover {

            background-color: #4D9096;

        }

        #predictButton:pressed {

            background-color: #438087;

        }


        /* ==========================================
           RESULT
        ========================================== */

        #resultArea {

            background-color: #F8FAFB;

            border: 1px solid #E5ECEE;

            border-radius: 15px;

            margin-top: 20px;

            margin-bottom: 20px;

        }

        #lowRiskArea {

            background-color: #F0F8F4;

            border: 1px solid #D5EBDD;

        }

        #highRiskArea {

            background-color: #FFF4F2;

            border: 1px solid #F1D7D2;

        }

        #resultIcon {

            background-color: #E7F1F3;

            color: #5A9FA5;

            border-radius: 28px;

            min-width: 56px;

            max-width: 56px;

            min-height: 56px;

            max-height: 56px;

            font-size: 28px;

            font-weight: 700;

        }

        #lowRiskIcon {

            background-color: #DFF1E8;

            color: #4E9B78;

            border-radius: 28px;

            min-width: 56px;

            max-width: 56px;

            min-height: 56px;

            max-height: 56px;

            font-size: 28px;

            font-weight: 700;

        }

        #highRiskIcon {

            background-color: #F8DED9;

            color: #C45A5A;

            border-radius: 28px;

            min-width: 56px;

            max-width: 56px;

            min-height: 56px;

            max-height: 56px;

            font-size: 28px;

            font-weight: 700;

        }

        #resultTitle {

            color: #2F4852;

            font-size: 22px;

            font-weight: 700;

            padding-top: 8px;

        }

        #resultText {

            color: #71878E;

            font-size: 11px;

            line-height: 1.5;

        }


        /* ==========================================
           PROBABILITY
        ========================================== */

        #smallTitle {

            color: #78909A;

            font-size: 10px;

            font-weight: 600;

        }

        #probability {

            color: #345963;

            font-size: 25px;

            font-weight: 700;

            padding: 4px;

        }


        /* ==========================================
           STATUS
        ========================================== */

        #statusLoaded {

            color: #4E9B78;

            background-color: #EDF8F3;

            border-radius: 9px;

            padding: 7px;

            font-size: 10px;

        }

        #statusError {

            color: #B66B62;

            background-color: #FFF3F1;

            border-radius: 9px;

            padding: 7px;

            font-size: 10px;

        }


        /* ==========================================
           DISCLAIMER
        ========================================== */

        #disclaimer {

            background-color: #FFF9F1;

            border: 1px solid #F0E3CC;

            border-radius: 11px;

            color: #826F54;

            padding: 11px;

            font-size: 10px;

        }

        """)


# =====================================================
# APPLICATION
# =====================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    app.setStyle(
        "Fusion"
    )

    window = HeartGuard()

    window.show()

    sys.exit(
        app.exec()
    )