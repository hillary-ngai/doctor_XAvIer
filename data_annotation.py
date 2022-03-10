import xml.etree.ElementTree as et
import ast
from collections import defaultdict
import glob
import string
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder


class MedDialogue:
    '''
    MedDialogue: object to help summarize dataset
    '''
    def __init__(self, data_folder='med-dialogue-data'):
        self.annotations_struct = defaultdict(list)
        self.data_folder = data_folder  # holds the xml dialogue files

    def set_annotation_struct(self):
        '''
        set_annotation_struct: sets the structure of each annotation. An example annotation structure is
        'sign_symptom': ['Type', 'descriptor', 'modality', 'pertinence', 'problem type', 'annotation_id', 'rawText']

        Returns: None
        '''
        annotation_struct = defaultdict(list)

        for file in glob.glob(self.data_folder + '/*/*.xml'):
            xmltree = et.parse(file)
            root = xmltree.getroot()
            transcripts = root.find('transcripts')
            annotations = transcripts.findall("annotations")
            annotations = ast.literal_eval(annotations[0].text)

            for annotation in annotations:
                if annotation[0] not in annotation_struct:
                    annotation_struct[annotation[0]] = list(annotation[1].keys())

        self.annotation_struct = annotation_struct

    def get_annotation_struct(self):
        '''
        get_annotation_struct: gets the structure of each annotation. An example annotation structure is

        Returns: defaultdict(list) self.annotation_struct
        '''
        return self.annotation_struct

    def NER_intent_df(self, split='train', save_df = False, filetype = 'csv',
                      save_destination = 'med-dialogue-data', filename = 'NER_med_dialogue_df.csv'):
        '''
        Args:
            split: either 'test' or 'train' dataset

        Returns: DataFrame with the columns filename, speaker_code, sequence, BIO_label, and intent
        '''
        all_filenames = []
        all_speaker_codes = []
        all_sequences = []
        all_entity_labels = []
        all_intents = []

        for file in glob.glob(self.data_folder + '/' + split + '/*.xml'):
            ta = TranscriptAnnotator(file)
            ta.annotate_entities()

            # for each sequence, join the BIO-labels into one string
            entity_labels = []
            for entity_label in ta.entity_labels:
                entity_labels.append(' '.join(entity_label))

            all_filenames += [os.path.basename(file)] * len(entity_labels)
            all_speaker_codes += ta.speaker_codes
            all_sequences += ta.sequences
            all_entity_labels += entity_labels
            all_intents += ta.intents

        df_dict = {
            'filename': all_filenames,
            'speaker_code': all_speaker_codes,
            'sequence': all_sequences,
            'entity_label': all_entity_labels,
            'intent': all_intents
        }

        df = pd.DataFrame(df_dict)
        df['sequence'].replace('', np.nan, inplace=True)
        df.dropna(inplace=True)
        df['prev_speaker_code'] = df['speaker_code'].shift(1)
        df['prev_sequence'] = df['sequence'].shift(1)
        df['prev_entity_label'] = df['entity_label'].shift(1)

        # save DataFrame
        if save_df:
            if filetype == 'pkl':
                df.to_pickle(save_destination + '/' + filename)
            elif filetype =='csv':
                df.to_csv(save_destination + '/' + filename, index=False)

        return df

    def patient_infos_df(self, split='test', save_df = False, filetype = 'csv',
                      save_destination = 'med-dialogue-data', filename = 'patient_infos_df.csv'):
        # map similar symptoms to one umbrella symptom
        symptom_map = {
            'paranoia': 'anxiety',
            'coughing': 'cough',
            'depressed mood': 'depression',
            'excessive alcohol drinking': 'excessive consumption of alcohol or use of chemical substances',
            'runny or clogged nose': 'runny / stuffy nose',
            'heart palpitations': 'sweating / palpitations',
            'joint aches, pain': 'joint pain',
            'joint tenderness': 'joint pain',
            'weight loss': 'weight loss or loss in appetite',
            'rapid breathing': 'difficulty breathing',
            'chr paain': 'chronic pain',
            'sob, edema': 'shortness of breath',
            'lethargy': 'low energy / fatigue',
            'patient did have probable with poor attention to detail, difficulty breathing responsibilities, fidgety, difficulty concentrating.':'difficulty in concentrating / distractibility',
            'unable to focus':'difficulty in concentrating / distractibility',
            'restlessness': 'anxiety',
            'vomiting': 'nausea',
            'dizziness': 'nausea',
            'agitation': 'irritability',
            'restlessness': 'anxiety',
            'night time awakenings': 'night symptoms',
            'wheezing': 'cough',
        }

        # group other medical conditions into more general terms
        other_medical_conditions_map = {
            'backache': 'back pain',
            'hs, vertigo': 'vertigo',
            'psoritic arthritis': 'arthritis',
            'psoriatic arthritis': 'arthritis',
            'rheumatoid arthritis': 'arthritis',
            'osteoarthritis': 'arthritis',
            'gout': 'arthritis',
            "alzheimer's disease": 'dementia',
            'atopic dermatitis,  h/o hcv': 'dermatitis',
            'atrial fibrillation,constipation': 'atrial fibrillation',
            'chronic colitis': 'ulcerative colitis',
            'lung nodule, cocci': 'cocci, lung nodule',
            'excessive underarm sweating': 'excessive sweating',
            'svt,excessive sweating': 'excessive sweating',
            'pph': '',
            'mv': '',
            'hs': '',
            'lfts': '',
            'hyperlipidemia, elevated bp, elevated liver enzymes': 'hyperlipoproteinemia',
            'hypercholesterolemia': 'hyperlipoproteinemia',
            'hypertriglyceridemia': 'hyperlipoproteinemia',
            'hypothyroidism': 'hypothyroid',
            'swollen gland, bradycardia, hsv': 'swollen gland',
            'swollen gland, pad, mvd, carotid dz': 'vascular disease',
            'thyroid dz, pad, caroitd dz': 'vascular disease',
            'thyroid dz, pad  czrotid dz': 'vascular disease',
            'pvd': 'vascular disease',
            'pvd, atelectasis': 'vascular disease',
            'pvd, pad, hypercalcemia': 'vascular disease',
            'a fib, pad, cll': 'vascular disease',
            'cad': 'heart disease',
            'irregular pulse': 'heart disease',
            'a fib, tingling, valve dz, carotid dz': 'vascular disease'
        }

        all_filename = []
        all_age = []
        all_gender = []
        all_height = []
        all_weight = []
        all_systolic_bp = []
        all_diastolic_bp = []
        all_currently_smokes = []
        all_employment_status = []
        all_reason_for_visit = []
        all_severity_of_disease = []
        all_other_medical_conditions = []
        all_symptoms = []
        all_treatments = []
        all_diagnosis = []

        for file in glob.glob(self.data_folder + '/' + split + '/*.xml'):
            ta = TranscriptAnnotator(file)
            diagnosis = ta.get_diagnosis()

            all_filename.append(os.path.basename(file))
            all_diagnosis.append(diagnosis)
            age = ''
            gender = ''
            height = ''
            weight = ''
            systolic_bp = ''
            diastolic_bp = ''
            currently_smokes = False # if not filled in, assume patient doesn't smoke
            reason_for_visit = ''
            severity_of_disease = 'not applicable'
            employment_status = ''
            other_medical_conditions = []
            symptoms = []
            treatments = []

            xmltree = et.parse(file)
            root = xmltree.getroot()
            field_responses = root.find("fieldResponses")
            for child, next_child in zip(field_responses, field_responses[1:]):
                if child.tag == 'field':
                    if child.text == 'Age (if less than 1, input 0)':
                        age = ''.join(filter(str.isdigit, next_child.text))
                        if age:
                            age = int(age)

                    elif child.text == 'Gender':
                        gender = next_child.text.lower()

                    elif child.text == 'Patient Height (inches)':
                        height = int(''.join(filter(str.isdigit, next_child.text)))
                        if height == 1:
                            height = None

                    elif child.text == 'Patient Weight (pounds)':
                        weight = int(''.join(filter(str.isdigit, next_child.text)))

                    elif child.text == 'Blood Pressure Systolic':
                        systolic_bp = int(''.join(filter(str.isdigit, next_child.text)))

                    elif child.text == 'Blood Pressure Diastolic':
                        diastolic_bp = int(''.join(filter(str.isdigit, next_child.text)))

                    elif child.text == 'Does the patient currently smoke?':
                        currently_smokes = next_child.text.lower() in ['true', '1', 't', 'y', 'yes']

                    elif child.text == 'Reason for today\'s visit' or child.text == 'Reason for today \'s visit':
                        reason_for_visit = next_child.text.lower()

                    elif child.text == 'Severity of disease':
                        severity_of_disease = next_child.text.lower()

                    elif child.text == 'Employment Status':
                        employment_status = next_child.text.lower()
                        if employment_status == 'don\'t know' or employment_status == 'not applicable':
                            employment_status = 'unknown'

                    elif child.text == 'Please select any other medical conditions the patient is suffering from (check all that apply)':
                        other_medical_condition = next_child.text.lower()

                        if other_medical_condition in other_medical_conditions_map.keys():
                            grouped_condition = other_medical_conditions_map[other_medical_condition]
                            if grouped_condition != '':
                                other_medical_conditions.append(grouped_condition)

                        else:
                            other_medical_conditions.append(other_medical_condition)

                    elif child.text == 'Please select the patient\'s current symptoms (check all that apply)':
                        symptom = next_child.text.lower()

                        if symptom in symptom_map.keys():
                            symptoms.append(symptom_map[symptom])

                        elif symptom != '1':
                            symptoms.append(symptom)

                    elif child.text == 'Treatment':
                        treatments.append(next_child.text.lower())

            all_other_medical_conditions.append(other_medical_conditions)
            all_symptoms.append(symptoms)
            all_treatments.append(treatments)

            all_age.append(age)
            all_gender.append(gender)
            all_height.append(height)
            all_weight.append(weight)
            all_systolic_bp.append(systolic_bp)
            all_diastolic_bp.append(diastolic_bp)
            all_currently_smokes.append(currently_smokes)
            all_reason_for_visit.append(reason_for_visit)
            all_severity_of_disease.append(severity_of_disease)
            all_employment_status.append(employment_status)

        patient_infos = {
            'filename': all_filename,
            'age': all_age,
            'gender': all_gender,
            'height': all_height,
            'weight': all_weight,
            'diastolic_bp': all_diastolic_bp,
            'systolic_bp': all_systolic_bp,
            'currently_smokes': all_currently_smokes,
            'employment_status': all_employment_status,
            'reason_for_visit': all_reason_for_visit,
            'severity_of_disease': all_severity_of_disease,
            'other_medical_conditions': all_other_medical_conditions,
            'symptoms': all_symptoms,
            'treatment': all_treatments,
            'diagnosis': all_diagnosis
        }

        df = pd.DataFrame(patient_infos)
        df.reset_index(drop=True, inplace=True)

        # convert to numeric values
        df['weight'] = pd.to_numeric(df['weight'])
        df['height'] = pd.to_numeric(df['height'])
        df['age'] = pd.to_numeric(df['age'])
        df['diastolic_bp'] = pd.to_numeric(df['diastolic_bp'])
        df['systolic_bp'] = pd.to_numeric(df['systolic_bp'])

        # fill empty values with the average value for each gender
        df['weight'] = df['weight'].fillna(df.groupby('gender')['weight'].transform('mean'))
        df['height'] = df['height'].fillna(df.groupby('gender')['height'].transform('mean'))
        df['age'] = df['age'].fillna(df.groupby('gender')['age'].transform('mean'))
        df['diastolic_bp'] = df['diastolic_bp'].fillna(df.groupby('gender')['diastolic_bp'].transform('mean'))
        df['systolic_bp'] = df['systolic_bp'].fillna(df.groupby('gender')['systolic_bp'].transform('mean'))

        if save_df:
            if filetype == 'pkl':
                df.to_pickle(save_destination + '/' + filename)
            elif filetype =='csv':
                df.to_csv(save_destination + '/' + filename, index=False)

        return df

    def one_hot_encoded_patient_infos_dfs(self, train_df, test_df, save_df=False, filetype='csv',
                                       save_destination = 'med-dialogue-data', df_name = 'encoded_patient_infos'):
        df = pd.concat([train_df, test_df])

        # encode symptoms with multi-label encoding
        symptoms_mlb = MultiLabelBinarizer()
        symptoms_mlb.fit(df['symptoms'])
        encoded_symptoms_train = pd.DataFrame(symptoms_mlb.transform(train_df['symptoms']),
                                              columns=['symptom_' + s for s in symptoms_mlb.classes_])
        encoded_symptoms_test = pd.DataFrame(symptoms_mlb.transform(test_df['symptoms']),
                                             columns=['symptom_' + s for s in symptoms_mlb.classes_])

        # encode other medical conditions with multi-label encoding
        conditions_mlb = MultiLabelBinarizer()
        conditions_mlb.fit(df['other_medical_conditions'])
        encoded_conditions_train = pd.DataFrame(conditions_mlb.transform(train_df['other_medical_conditions']),
                                          columns=['other_medical_conditions_' + s for s in conditions_mlb.classes_])
        encoded_conditions_test = pd.DataFrame(conditions_mlb.transform(test_df['other_medical_conditions']),
                                                columns=['other_medical_conditions_' + s for s in
                                                         conditions_mlb.classes_])

        # one-hot encode employment status
        employment_enc = OneHotEncoder()
        employment_enc.fit(df[['employment_status']])

        one_hot_employment_train = pd.DataFrame(employment_enc.transform(train_df[['employment_status']]).toarray(),
                                               columns=['employment_status_' + s for s in employment_enc.categories_[0]])
        one_hot_employment_test = pd.DataFrame(employment_enc.transform(test_df[['employment_status']]).toarray(),
                                               columns=['employment_status_' + s for s in employment_enc.categories_[0]])

        # one-hot encode reason_for_visit
        visit_reason_enc = OneHotEncoder()
        visit_reason_enc.fit(df[['reason_for_visit']])
        one_hot_visit_reason_train = pd.DataFrame(visit_reason_enc.transform(train_df[['reason_for_visit']]).toarray(),
                                                columns=['reason_for_visit_' + s for s in visit_reason_enc.categories_[0]])
        one_hot_visit_reason_test = pd.DataFrame(visit_reason_enc.transform(test_df[['reason_for_visit']]).toarray(),
                                               columns=['reason_for_visit_' + s for s in visit_reason_enc.categories_[0]])

        # combine encodings in one dataframe
        train_df = pd.concat([
                train_df[['filename', 'age', 'gender', 'height', 'weight', 'diastolic_bp', 'systolic_bp',
                          'currently_smokes']],
                one_hot_employment_train,
                one_hot_visit_reason_train,
                encoded_symptoms_train,
                encoded_conditions_train,
                train_df['diagnosis']
            ],
            axis = 1)

        test_df = pd.concat([
            test_df[['filename', 'age', 'gender', 'height', 'weight', 'diastolic_bp', 'systolic_bp',
                      'currently_smokes']],
            one_hot_employment_test,
            one_hot_visit_reason_test,
            encoded_symptoms_test,
            encoded_conditions_test,
            test_df['diagnosis']
            ],
            axis=1)

        # map gender
        gender_map = {
            'female': 1,
            'male': 0
        }
        train_df['gender'] = train_df['gender'].map(gender_map)
        test_df['gender'] = test_df['gender'].map(gender_map)

        # map diagnosis
        diagnosis_map = {
            'adhd': 0,
            'depression': 1,
            'osteoporosis': 2,
            'influenza': 3,
            'copd': 4,
            'type ii diabetes': 5,
            'other': 6
        }
        train_df['diagnosis'] = train_df['diagnosis'].map(diagnosis_map)
        test_df['diagnosis'] = test_df['diagnosis'].map(diagnosis_map)

        if save_df:
            if filetype == 'pkl':
                train_df.to_pickle(save_destination + '/' + df_name + '_train.csv')
                test_df.to_pickle(save_destination + '/' + df_name + '_test.csv')
            elif filetype =='csv':
                train_df.to_csv(save_destination + '/' + df_name + '_train.csv', index=False)
                test_df.to_csv(save_destination + '/' + df_name + '_test.csv', index=False)

        return train_df, test_df

    def diagnosis_symptom_df(self, save_df=False, filetype='csv',
                         save_destination='med-dialogue-data', filename='diagnosis_symptom_df.csv'):
        diagnoses = []
        symptoms = []

        for file in glob.glob(self.data_folder + '/*/*.xml'):
            ta = TranscriptAnnotator(file)
            diagnosis = ta.get_diagnosis()

            annotations = ta.get_annotations()
            speaker_codes, sequences = ta.get_sequences()
            for annotation in annotations:
                sequence_id = annotation[2][0][0]
                sequence = sequences[sequence_id].split()  # full sequence
                speaker_code = speaker_codes[sequence_id]  # Doctor (DR) vs. Patient (PT)

                # only the annotated portion of the sequence
                start_idx = annotation[2][0][1]  # start char index
                end_idx = annotation[2][-1][2]  # end char index
                annotated_sequence = sequences[sequence_id][start_idx:end_idx].split()  # list of words

                # remove punctuation, make lowercase, still keep original string split
                sequence = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in sequence]
                annotated_sequence = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in
                                      annotated_sequence]

                annotation_type = annotation[0]

                # get the patient's self-disclosed symptoms
                if annotation_type == 'sign_symptom' and 'modality' in annotation[1].keys() \
                        and annotation[1]['modality'] == 'actual':
                    symptom = ' '.join(annotated_sequence)
                    if symptom:
                        diagnoses.append(diagnosis)
                        symptoms.append(symptom)

        diagnosis_symptom_dict = {
            'diagnosis': diagnoses,
            'symptom': symptoms
        }
        df = pd.DataFrame(diagnosis_symptom_dict)

        # save DataFrame
        if save_df:
            if filetype == 'pkl':
                df.to_pickle(save_destination + '/' + filename)
            elif filetype == 'csv':
                df.to_csv(save_destination + '/' + filename, index=False)

        return diagnosis_symptom_dict


class TranscriptAnnotator:
    '''
    TranscriptAnnotator: object to store annotated transcript
    '''
    def __init__(self, file_path=''):
        self.file_path = file_path
        self.sequences = None # list of string sequences
        self.speaker_codes = None # list of string speaker_codes
        self.BIO_labels = None # a list of list of BIO-labels (first dimension corresponds to sequence index)
        self.intents = None
        self.entity_labels = None # a list of list of BIO-labels (first dimension corresponds to sequence index)
        self.diagnosis = None

        # for NER
        self.entity_map = {
            'diagnosis': 'diagnosis',
            'sign_symptom': 'sign_symptom',
            'referral': 'referral',
            'TIMEX3': 'time_expression',
            'timex3': 'time_expression',
            'investigation_therapy': 'investigation_therapy',
            'medication': 'medication',
            'bodily_function_vital_sign': 'bodily_function_vital_sign',
            'anatomical_locations_macroscopic_microscopic': 'anatomical_locations',
            'substance_use': 'substance_use',
            'allergy_intolerance': 'allergy_intolerance',
            'immunizations': 'immunizations'
        }

    def set_diagnosis(self):
        # map diagnosis
        diagnosis_map = {
            'adult adhd': 'adhd',
            'well visit': 'other',
            'atopic dermatitis': 'other',
            'atrial fibrillation': 'other',
            'uterine fibroids': 'other',
            'hypercholesterolemia': 'other',
            'chf': 'other',
            'hidradenitis suppurativa': 'other',
            'venous thrombo-embolism': 'other',
            'asthma': 'other'
        }

        xmltree = et.parse(self.file_path)
        root = xmltree.getroot()

        self.diagnosis = root.find('interactionType').text.lower()
        if self.diagnosis in diagnosis_map.keys():
            self.diagnosis = diagnosis_map[self.diagnosis]

    def get_diagnosis(self):
        if not self.diagnosis:
            self.set_diagnosis()

        return self.diagnosis

    def set_sequences(self):
        '''
        set_sequences: sets transcript sequences as list string speaker_codes (e.g. DR = doctor) sequences and a list
        of string sequences (e.g. "How's your appetite?"). Rows correspond to sequence index (starting from 1).

        Args:
            file_path: XML file path of transcript

        Returns: None
        '''
        xmltree = et.parse(self.file_path)
        root = xmltree.getroot()

        # sequences stores a list of strings
        sequences = [''] # start with dummy variable since the sequences indices start at 1
        speaker_codes = [''] # start with dummy variable since the sequences indices start at 1

        transcripts = root.find('transcripts')
        turns = transcripts.find('turns')

        for utterance in turns.findall('utterance'):
            if 'speakerCode' in utterance.attrib.keys():
                speaker_codes.append(utterance.attrib['speakerCode'])

            # if we can't find the speaker code, append a dummy variable
            else:
                speaker_codes.append('unknown_speaker')

            sequences.append(utterance.text)

        self.sequences = sequences
        self.speaker_codes = speaker_codes

    def get_sequences(self):
        '''
        Returns: transcript sequences as list of string speaker codes and list of string sequences
        '''
        if not self.sequences:
            self.set_sequences()

        return self.speaker_codes, self.sequences

    def get_annotations(self):
        '''
        Returns: annotations as a list
        '''
        xmltree = et.parse(self.file_path)
        root = xmltree.getroot()
        transcripts = root.find('transcripts')
        annotations = transcripts.findall("annotations")
        annotations = ast.literal_eval(annotations[0].text)

        return annotations

    def annotate_entities(self):
        '''
        annotate_transcript: annotates the sequences as a list of list of BIO-labels and sets BIO_labels.

        Returns: None
        '''
        speaker_codes, sequences = self.get_sequences()
        annotations = self.get_annotations()

        self.entity_labels = []
        # start by labelling everything as 'O' for outside
        for sequence in sequences:
            entity_label = ['O' for word in sequence.split()]
            self.entity_labels.append(entity_label)

        # intents
        self.intents = ['other'] * len(sequences)

        # use last PT sequence as 'closing' from last three sequences
        i = len(sequences) - 1
        while i >= len(sequences) - 3:
            last_speaker_code = speaker_codes[i]

            if last_speaker_code == 'PT' or last_speaker_code == 'CG':
                self.intents[i] = 'closing'
                break

            i -= 1

        # use annotations for the rest of the BIO-labels
        for annotation in annotations:
            sequence_id = annotation[2][0][0]
            sequence = sequences[sequence_id].split()  # full sequence
            speaker_code = speaker_codes[sequence_id]  # Doctor (DR) vs. Patient (PT)

            # only the annotated portion of the sequence
            start_idx = annotation[2][0][1]  # start char index
            end_idx = annotation[2][-1][2]  # end char index
            annotated_sequence = sequences[sequence_id][start_idx:end_idx].split()  # list of words

            # remove punctuation, make lowercase, still keep original string split
            sequence = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in sequence]
            annotated_sequence = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in
                                  annotated_sequence]

            annotation_start_idx = 0  # word start index for annotation
            for i in range(len(sequence)):
                # sliding window to check if they are the same list and find the first word index
                if sequence[i:i + len(annotated_sequence)] == annotated_sequence:
                    annotation_start_idx = i
                    break

            annotation_type = annotation[0]
            # label with BIO-labels
            if annotation_type in self.entity_map.keys():
                self._annotate_entities_in_sequence(annotation_start_idx, len(annotated_sequence),
                                        sequence_id, annotation_type)

            # special case where we also want to label the reply of the patient
            if annotation_type == 'sign_symptom' and (speaker_code == 'DR' or speaker_code == 'NR'):
                if sequence_id + 1 < len(sequences):
                    next_sequence = sequences[sequence_id + 1].split()  # full sequence
                    next_speaker_code = speaker_codes[sequence_id + 1]

                    if 'modality' in annotation[1].keys() and (next_speaker_code == 'PT' or next_speaker_code == 'CG'):
                        symptom_modality = annotation[1]['modality']

                        # label whether or not the patient confirmed/denied/is unsure of the symptom
                        if symptom_modality == 'actual':
                            self.intents[sequence_id + 1] = 'confirm+symptom'

                        elif symptom_modality == 'negative':
                            self.intents[sequence_id + 1] = 'deny+symptom'

                        else:
                            self.intents[sequence_id + 1] = 'unsure+symptom'

            # label intent of patient
            if speaker_code == 'PT' or speaker_code == 'CG':
                # label when the patient self-discloses symptoms
                if annotation_type == 'sign_symptom' and 'modality' in annotation[1].keys():
                        # patient self-discloses symptoms
                        if annotation[1]['modality'] == 'actual':
                            self.intents[sequence_id] = 'confirm+symptom'

                        # they know they don't have this symptom
                        elif annotation[1]['modality'] == 'negative':
                            self.intents[sequence_id] = 'deny+symptom'

                        # unsure about the symptom that they are self-disclosing
                        else:
                            self.intents[sequence_id] = 'unsure+symptom'


    def _annotate_entities_in_sequence(self, annotation_start_idx, len_annotated_sequence, sequence_id,
                                       annotation_type):
        '''
        _annotate_sequence: helper funtion to annotate sequence

        Args:
            annotation_start_idx: char start index of annotation in the sequence
            len_annotated_sequence: number of chars in the annotated sequence
            sequence_id: sequence ID
            speaker_code: who is speaking (e.g. DR, PT, etc.)
            annotation_type: annotation type (e.g. 'sign_symptom', 'reason_for_visit', etc.)

        Returns: None

        '''
        j = annotation_start_idx

        while j < annotation_start_idx + len_annotated_sequence:
            # beginning word
            if j == annotation_start_idx:
                self.entity_labels[sequence_id][j] = 'B-' + self.entity_map[annotation_type]
            # inside word
            else:
                self.entity_labels[sequence_id][j] = 'I-' + self.entity_map[annotation_type]

            j += 1



md = MedDialogue(data_folder='med-dialogue-data')
#md.set_annotation_struct()
#print(md.get_annotation_struct().keys())
'''test_df = md.patient_infos_df(split='test', save_df=True, filetype='csv',
                          save_destination = 'med-dialogue-data', filename = 'patient_infos_test.csv')
train_df = md.patient_infos_df(split='train', save_df=True, filetype='csv',
                          save_destination = 'med-dialogue-data', filename = 'patient_infos_train.csv')
print(md.one_hot_encoded_patient_infos_dfs(train_df, test_df, save_df=True, save_destination = 'med-dialogue-data'))'''


print(md.diagnosis_symptom_df(save_df=True, filetype='csv',
                         save_destination='med-dialogue-data', filename='diagnosis_symptom_df.csv'))