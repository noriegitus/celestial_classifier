
-- Complete Cleaning
-- Deleting tables in their order of creation to avoid dependency issues
-- CASCADE assures that all dependencies are deleted as well
DROP TABLE IF EXISTS Prediction CASCADE;
DROP TABLE IF EXISTS Evaluation CASCADE;
DROP TABLE IF EXISTS Image CASCADE;
DROP TABLE IF EXISTS Model CASCADE;

-- Create type ENUM for labels
DROP TYPE IF EXISTS galaxy_class;
CREATE TYPE galaxy_class AS ENUM ('elliptical','spiral','irregular');

----------------------------------------------
-- TABLE 1: Models
-- Storage info about each architecture and version of the model trained
CREATE TABLE Model(
	model_id 		SERIAL PRIMARY KEY,
	architecture 	VARCHAR(50) NOT NULL, -- "CNN" or "ResNet18"
	version 		VARCHAR(20) NOT NULL, -- "v2", "v3"
	description 	TEXT
);

----------------------------------------------
-- TABLE 2: Image
-- Central catalogue of all images, with their metadata and true labels
CREATE TABLE Image (
    image_id              SERIAL PRIMARY KEY,
    filename              VARCHAR(255) UNIQUE NOT NULL,
    true_label            galaxy_class NOT NULL,
    description           TEXT,
    folder                VARCHAR(50),
    source                VARCHAR(50),
    avg_brightness        NUMERIC(8,4),
    avg_contrast          NUMERIC(8,4),
    dom_color_hex         VARCHAR(7),
    height_px             INTEGER,
    width_px              INTEGER,
    filesize_kb           INTEGER
);


----------------------------------------------
-- TABLE 3: Evaluation
-- Storages final performance metrics for each model
CREATE TABLE Evaluation(
	evaluation_id 	SERIAL PRIMARY KEY,
	model_id 		INT NOT NULL REFERENCES Model(model_id) ON DELETE CASCADE,
    accuracy		NUMERIC(5, 4),
    precision 		NUMERIC(5, 4),
    recall 			NUMERIC(5, 4),
    f1_score 		NUMERIC(5, 4)
);

----------------------------------------------
-- TABLE 3: Prediction
-- Info on each individual prediction for later error analysis
CREATE TABLE Prediction (
    prediction_id     SERIAL PRIMARY KEY,
    image_id          INT NOT NULL REFERENCES Image(image_id) ON DELETE CASCADE,
    model_id          INT NOT NULL REFERENCES Model(model_id) ON DELETE CASCADE,
    predicted_label   galaxy_class,
    is_correct        BOOLEAN,
    confidence        NUMERIC(10,9)
);

----------------------------------------------
-- INDEXING FOREIGN KEYS 
-- For higher reading speeds on querys from PowerBi
CREATE INDEX idx_evaluations_model ON Evaluation(model_id);
CREATE INDEX idx_predictions_image ON Prediction(image_id);
CREATE INDEX idx_predictions_model ON Prediction(model_id);

SELECT 'Database Created Successfully!' as status;