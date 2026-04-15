# The Practitioner’s Guide to Classification for Human Kinematics

This guide outlines the technical workflow for transforming raw, high-dimensional motion capture data into a predictive classification model. When analyzing dyadic interactions (two participants) to determine a binary outcome (e.g., Success/Failure or Winner/Loser), we must bridge the gap between temporal sequence data and static supervised learning estimators.

---

## 1. Spatial Normalization and Invariance
**The Objective:** Transform raw x,y coordinate data into a participant-centric and scale-independent representation.
**The Rationale:** Raw coordinates provided by computer vision models are "view-dependent," meaning they are influenced by the camera’s distance, angle, and field of view. To ensure the model learns kinematic patterns rather than spatial positioning, we apply three layers of normalization:

* **Translation:** Centering a "root" joint (typically the mid-hip) at the origin (0,0). This eliminates "global position" bias, focusing the model exclusively on relative limb movement.
* **Scaling:** Dividing coordinates by a reference skeletal length (e.g., torso height). This ensures that the physical stature of a participant—or their distance from the lens—does not introduce artificial variance into the dataset.
* **Angular Transformation:** Deriving joint angles (e.g., knee flexion, elbow extension) from coordinate triplets. Because angles are properties of the vectors between joints, they are inherently invariant to both translation and scaling. This provides the model with "pose-primitive" features that remain constant across different camera setups and body types.

---

## 2. Temporal Feature Engineering
**The Objective:** Derive higher-order physical properties (derivatives) from positional data.
**The Rationale:** Static coordinates at any single frame t lack sufficient context for intent or force. To capture the "mechanics" of the movement, we calculate:
* **Kinematic Derivatives:** We compute first-order (velocity) and second-order (acceleration) derivatives. These represent the intensity and explosive nature of the movement.
* **Inter-agent Proxemics:** Calculating the Euclidean distance between the two participants to capture the spatial pressure and timing of the interaction.

---

## 3. Statistical Dimensionality Reduction (Aggregation)
**The Objective:** Compress variable-length time-series data into a fixed-width feature vector.
**The Rationale:** Standard supervised estimators (e.g., Support Vector Machines, Random Forests) require a consistent input dimension. However, human movements are inherently variable in duration. 
By applying statistical moments—Mean, Standard Deviation, Max, Skewness, and Kurtosis—to our kinematic features, we collapse the temporal dimension. This creates a "Kinematic Fingerprint" for each participant that summarizes their performance across the entire 2–4 second window.

---

## 4. Feature Scaling and Standardization
**The Objective:** Align the magnitudes of disparate feature types using Z-score normalization.
**The Rationale:** Our feature set contains units with vastly different scales (e.g., degrees for angles vs. pixels/second for velocity). Many ML algorithms use distance-based calculations or gradient descent. Without scaling, features with larger raw values would exert a disproportionate influence on the model’s weight updates, effectively "drowning out" subtle but critical indicators.

---

## 5. The Ensemble Classifier (Random Forest)
**The Objective:** Utilize a robust, non-linear estimator to map features to outcomes.
**The Rationale:** Human movement is non-linear and involves complex interactions between variables. The Random Forest algorithm is ideal because:
* **Non-linearity:** It can capture complex decision boundaries that linear models might miss.
* **Feature Robustness:** It is less sensitive to outliers and does not assume a normal distribution of data.
* **Interpretability:** It allows us to calculate Feature Importance, revealing which biomechanical markers are the primary drivers of the outcome.

---

## 6. Group-Based Validation (Preventing Data Leakage)
**The Objective:** Implement a validation strategy that respects the dependency between opponents.
**The Rationale:** In any competitive interaction, the performance of Participant A is intrinsically linked to Participant B. If we randomly split individual rows into training and testing sets, the model may encounter one participant from Match #1 during training and their opponent during testing. This results in Data Leakage. We use GroupKFold, ensuring that all data associated with a specific match_ID is strictly partitioned into either the training or testing set, never both.

---

## 7. Model Evaluation and Interpretability
**The Objective:** Quantify performance and extract domain insights.
**The Rationale:** Beyond simple accuracy, we analyze the model's Confusion Matrix to identify if it is biased toward predicting winners or losers. Finally, we examine the Gini Importance of our features. This transitions the project from a "Black Box" prediction tool to a diagnostic tool that identifies the specific movements that correlate with success.
