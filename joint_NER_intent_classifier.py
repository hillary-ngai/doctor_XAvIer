import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
import torch.nn.functional as F
import argparse
import os
from transformers import AutoTokenizer, AutoModel


# PyTorch Dataset object
class MedDialogueDataset(Dataset):
    def __init__(self, df, entity_labels, tokenizer, max_len):
        self.sequences = df['sequence']
        self.entity_labels = entity_labels
        self.intents = df['intent']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        sequence = str(self.sequences[item])
        entity_label = self.entity_labels[item]
        intent = self.intents[item]

        encoding = self.tokenizer.encode_plus(
            sequence,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )

        return {
            'sequence': sequence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'entity_label': torch.tensor(entity_label, dtype=torch.long),
            'intent_label': torch.tensor(intent, dtype=torch.long)
        }


def _map_intent(df):
    intent_map = {
        'confirm+symptom': 0,
        'deny+symptom': 1,
        'unsure+symptom': 2,
        'closing': 3,
        'other': 4
    }

    df['intent'] = df['intent'].map(intent_map)

    return df


def _combine_prev_sequences_entities(df):
    df['sequence'] = df['prev_speaker_code'] + ': ' + df['prev_sequence'] + ' ' + df['speaker_code'] + ': ' + df[
        'sequence']
    df['entity_label'] = 'O ' + df['prev_entity_label'] + ' O ' + df['entity_label']

    return df


def preprocess_data(tokenizer, train_file, test_file, max_len):
    # read data frames
    train_df = pd.read_csv(train_file)
    train_df.fillna('', inplace=True)
    test_df = pd.read_csv(test_file)
    test_df.fillna('', inplace=True)

    # combine previous sequences and entitie labels
    train_df = _combine_prev_sequences_entities(train_df)
    test_df = _combine_prev_sequences_entities(test_df)

    # map intents to integers
    train_df = _map_intent(train_df)
    test_df = _map_intent(test_df)

    # drop NaNs
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # reset index
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # map entities to integers
    entity_labels = set()
    for BIO_entity_labels in train_df['entity_label']:
        for entity_label in BIO_entity_labels.split():
            entity_labels.add(entity_label)

    slot_names = ["[PAD]"]
    slot_names += entity_labels
    entity_map = dict((entity, idx) for idx, entity in enumerate(slot_names))

    # encode tokens with integer entity labels (e.g. [PAD]: 0, 'Ibuprofin': 1 ('B-medication'))
    entity_test = _encode_token_labels(test_df['sequence'],
                                       test_df['entity_label'], tokenizer, entity_map, max_len)

    entity_train = _encode_token_labels(train_df['sequence'],
                                        train_df['entity_label'], tokenizer, entity_map, max_len)

    return train_df, test_df, entity_train, entity_test, entity_map


def create_data_loader(df, entity_labels, tokenizer, max_len, batch_size):
    ds = MedDialogueDataset(
        df=df,
        entity_labels=entity_labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )


def _encode_token_labels(sequences, all_BIO_labels, tokenizer, slot_map,
                         max_length=128):
    encoded = np.zeros(shape=(len(sequences), max_length), dtype=np.int32)

    for i, (sequence, BIO_labels) in enumerate(zip(sequences, all_BIO_labels)):
        encoded_labels = []

        for word, BIO_label in zip(sequence.split(), BIO_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[BIO_label])

            expand_label = BIO_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = BIO_label

            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))

        encoded[i, 1:min(len(encoded_labels) + 1, max_length)] = encoded_labels[:max_length - 1]

    return encoded


# named entity recognition classifier
class NERIntentClassifier(nn.Module):
    def __init__(self, num_entity_labels, num_intent_labels,
                 pretrained_model_name='emilyalsentzer/Bio_ClinicalBERT',
                 dropout_prob=0.1):
        super(NERIntentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name,
                                              output_hidden_states=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.NER_classifier = nn.Linear(self.bert.config.hidden_size * 4,
                                        num_entity_labels)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size * 4,
                                           num_intent_labels)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # concatenate BERT's last 4 hidden states
        hidden_states = bert_output['hidden_states']
        last_hidden_states = torch.cat(
            tuple([hidden_states[i] for i in [-4, -3, -2, -1]]),
            dim=-1
        )

        # ------ NER Classification ------
        # dim: batch_size x batch_max_len x (bert_hidden_dim*4)
        last_hidden_states = self.dropout(last_hidden_states)

        # reshape the Variable so that each row contains one token
        # dim: batch_size x batch_max_len x (bert_hidden_dim*4)
        NER_hidden_state = last_hidden_states.view(-1, last_hidden_states.shape[2])

        # dim: batch_size*batch_max_len x num_entity_labels
        entity_logits = self.NER_classifier(NER_hidden_state)

        # ------ Intent Classification ------
        # pool output for sequence classification
        # only use [CLS] token embedding
        pooled_output = last_hidden_states[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # dim: (batch_size * 1) x num_intent_labels
        intent_logits = self.intent_classifier(pooled_output)

        return F.log_softmax(entity_logits, dim=1), F.log_softmax(intent_logits, dim=1)


# write custom loss function to exclude [PAD] tokens
def joint_loss_fn(NER_outputs, intent_outputs, entity_labels, intent_labels,
                  intent_loss_fn, alpha=0.5):
    # ------ NER loss ------
    # reshape labels to give a flat vector of length batch_size*seq_len
    entity_labels = entity_labels.view(-1)

    # mask out '[PAD]' tokens
    mask = (entity_labels > 0).float()

    # We then compute the Negative Log Likelihood Loss
    # (remember the output from the network is already softmax-ed and log-ed!)
    # for all the non PAD tokens

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data)

    # pick the values corresponding to labels and multiply by mask
    masked_output = NER_outputs[range(NER_outputs.shape[0]), entity_labels] * mask

    # NER cross entropy loss using Monte Carlo estimation
    NER_cross_entropy_loss = -torch.sum(masked_output) / num_tokens

    # ------ Intent loss ------
    intent_cross_entropy_loss = intent_loss_fn(intent_outputs, intent_labels)

    return (1 - alpha) * NER_cross_entropy_loss + alpha * intent_cross_entropy_loss


# write custom acc function to exclude [PAD] tokens
def _NER_num_masked_correct_preds_num_tokens(outputs, entity_labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    entity_labels = entity_labels.view(-1)

    # mask out '[PAD]' tokens
    mask = (entity_labels > 0).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data)

    preds = torch.argmax(outputs, dim=1)

    masked_correct_preds = (preds == entity_labels) * mask

    return torch.sum(masked_correct_preds), num_tokens


# write custom acc function to exclude [PAD] tokens
def _NER_masked_preds_tokens(outputs, entity_labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    entity_labels = entity_labels.view(-1)

    # mask out '[PAD]' tokens
    mask = (entity_labels > 0).float()

    preds = torch.argmax(outputs, dim=1)

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data)

    return preds, entity_labels, num_tokens


def train_model(model, data_loader, intent_loss_fn, optimizer,
                device, scheduler):
    model = model.train()
    losses = []  # used to calculate running average batch loss
    NER_total_correct_preds = 0  # total correct NER masked predictions for the epoch (entire data)
    total_num_tokens = 0  # total number of tokens after masking out '[PAD]' tokens
    intent_correct_preds = 0  # total correct intent predictions for the epoch
    total_num_sequences = 0

    for i, d in enumerate(data_loader):
        # get the inputs and labels
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        entity_labels = d['entity_label'].to(device)
        intent_labels = d['intent_label'].to(device)

        # zero out the gradients since
        # PyTorch accumulates the gradients on subsequent backward passes
        optimizer.zero_grad()

        # forward pass
        NER_outputs, intent_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # calculate negative log-likelihood loss
        loss = joint_loss_fn(NER_outputs, intent_outputs,
                             entity_labels, intent_labels,
                             intent_loss_fn, alpha=0.5).to(device)
        losses.append(loss.item())

        # backward + optimize
        loss.backward()  # compute gradients
        # avoiding exploding gradients by clipping the gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()  # update parameters
        scheduler.step()  # decays learning rate by gamma every step

        # ------ NER accuracy ------
        # count number of correct predictions and ...
        # total number of tokens after masking out the '[PAD]' labels
        NER_correct_preds, num_tokens = _NER_num_masked_correct_preds_num_tokens(
            NER_outputs, entity_labels
        )
        NER_total_correct_preds += NER_correct_preds
        total_num_tokens += num_tokens

        # ------ Intent accuracy ------
        intent_preds = torch.argmax(intent_outputs, dim=1)
        intent_correct_preds += torch.sum(intent_preds == intent_labels)
        total_num_sequences += len(d['sequence'])

        if i % 50 == 0:
            print(f'NER Train Accuracy: {NER_total_correct_preds / total_num_tokens}')
            print(f'Intent Train Accuracy: {intent_correct_preds / total_num_sequences}')
            print(f'Train Loss: {np.mean(losses)}')
            print('')


def validate_model(model, data_loader, intent_loss_fn, device):
    model = model.eval()
    losses = []  # used to calculate running average batch loss
    NER_total_correct_preds = 0  # total correct NER masked predictions for the epoch (entire data)
    total_num_tokens = 0  # total number of tokens after masking out '[PAD]' tokens
    intent_correct_preds = 0  # total correct intent predictions for the epoch
    total_num_sequences = 0

    with torch.no_grad():
        for i, d in enumerate(data_loader):
            # get the inputs and labels
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            entity_labels = d['entity_label'].to(device)
            intent_labels = d['intent_label'].to(device)

            # forward pass
            NER_outputs, intent_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # calculate negative log-likelihood loss
            loss = joint_loss_fn(NER_outputs, intent_outputs,
                                 entity_labels, intent_labels,
                                 intent_loss_fn, alpha=0.5).to(device)
            losses.append(loss.item())

            # ------ NER accuracy ------
            # count number of correct predictions and ...
            # total number of tokens after masking out the '[PAD]' labels
            NER_correct_preds, num_tokens = _NER_num_masked_correct_preds_num_tokens(
                NER_outputs, entity_labels
            )
            NER_total_correct_preds += NER_correct_preds
            total_num_tokens += num_tokens

            # ------ Intent accuracy ------
            intent_preds = torch.argmax(intent_outputs, dim=1)
            intent_correct_preds += torch.sum(intent_preds == intent_labels)
            total_num_sequences += len(d['sequence'])

    print(f'NER Validation Accuracy: {NER_total_correct_preds / total_num_tokens}')
    print(f'Intent Validation Accuracy: {intent_correct_preds / total_num_sequences}')
    print(f'Validation Loss: {np.mean(losses)}')
    print('')


def predict(model, data_loader, intent_loss_fn, device, out_dir):
    model = model.eval()
    losses = []  # used to calculate running average batch loss
    NER_total_correct_preds = 0  # total correct NER masked predictions for the epoch (entire data)
    total_num_tokens = 0  # total number of tokens after masking out '[PAD]' tokens
    intent_correct_preds = 0  # total correct intent predictions for the epoch
    total_num_sequences = 0
    all_token_predictions = []
    all_intent_predictions = []
    all_token_targets = []
    all_intent_targets = []

    with torch.no_grad():
        for i, d in enumerate(data_loader):
            # get the inputs and labels
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            entity_labels = d['entity_label'].to(device)
            intent_labels = d['intent_label'].to(device)

            # forward pass
            NER_outputs, intent_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # calculate negative log-likelihood loss
            loss = joint_loss_fn(NER_outputs, intent_outputs,
                                 entity_labels, intent_labels,
                                 intent_loss_fn, alpha=0.5).to(device)
            losses.append(loss.item())

            # ------ NER accuracy ------
            # count number of correct predictions and ...
            # total number of tokens after masking out the '[PAD]' labels
            token_predictions, token_targets, num_tokens = _NER_masked_preds_tokens(NER_outputs, entity_labels)
            mask = (entity_labels.view(-1) > 0).float()
            NER_correct_preds = torch.sum((token_predictions == token_targets) * mask)
            NER_total_correct_preds += NER_correct_preds
            total_num_tokens += num_tokens

            # ------ Intent accuracy ------
            intent_preds = torch.argmax(intent_outputs, dim=1)
            intent_correct_preds += torch.sum(intent_preds == intent_labels)
            total_num_sequences += len(d['sequence'])

            # save predictions
            all_token_predictions.extend(token_predictions.data.cpu().tolist())
            all_token_targets.extend(token_targets.data.cpu().tolist())

            all_intent_predictions.extend(intent_preds.data.cpu().tolist())
            all_intent_targets.extend(intent_labels.data.cpu().tolist())

    NER_results_dict = {
        'all_token_predictions': all_token_predictions,
        'all_token_targets': all_token_targets
    }

    intent_results_dict = {
        'all_intent_predictions': all_intent_predictions,
        'all_intent_targets': all_intent_targets
    }

    NER_results_df = pd.DataFrame(NER_results_dict)
    NER_results_df.to_csv(out_dir + '/NER_results_df.csv', index=False)

    intent_results_df = pd.DataFrame(intent_results_dict)
    intent_results_df.to_csv(out_dir + '/intent_results_df.csv', index=False)

    print(f'NER Total Tokens: {total_num_tokens}')
    print(f'NER Test Accuracy: {NER_total_correct_preds / total_num_tokens}')
    print(f'Intent Test Accuracy: {intent_correct_preds / total_num_sequences}')
    print(f'Test Loss: {np.mean(losses)}')
    print('')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='NER Intent BERT Classifier')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--max-len', type=int, default=128, metavar='ML',
                        help='maximum sequence length (default: 128)')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS',
                        help='batch size (default: 16)')
    parser.add_argument('--train-file', type=str, default='med-dialogue-data/md_train_df.csv', metavar='TR',
                        help='train CSV file (default: med-dialogue-data/md_train_df.csv)')
    parser.add_argument('--test-file', type=str, default='med-dialogue-data/md_test_df.csv', metavar='TE',
                        help='train CSV file (default: med-dialogue-data/md_test_df.csv)')
    parser.add_argument('--out-dir', type=str, default='output', metavar='O',
                        help='output directory of TensorBoard logs (default: output)')
    parser.add_argument('--pretrained-model-name', type=str, default='emilyalsentzer/Bio_ClinicalBERT', metavar='P',
                        help='pretrained model name (default: emilyalsentzer/Bio_ClinicalBERT)')
    parser.add_argument('--NER-intent-model-name', type=str, default='NER_intent_BERT_classifier_state_dict.pt', metavar='P',
                        help='NER and intent classifier model name (default: NER_intent_BERT_classifier_state_dict.pt)')
    args = parser.parse_args()
    epoch = 0

    # load checkpoint
    checkpoint_path = os.path.join(args.out_dir, args.NER_intent_model_name)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        args = state_dict['args']
        model_state_dict = state_dict['model_state_dict']

    # Preprocess data
    # BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    train_df, test_df, entity_train, entity_test, entity_map = preprocess_data(tokenizer, args.train_file,
                                                                               args.test_file, args.max_len)

    # create train and test data loader objects
    train_data_loader = create_data_loader(train_df, entity_train, tokenizer, args.max_len, args.batch_size)
    test_data_loader = create_data_loader(test_df, entity_test, tokenizer, args.max_len, args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NERIntentClassifier(len(entity_map), len(train_df['intent'].unique()),
                                pretrained_model_name=args.pretrained_model_name)

    # load model state dict from where you left off training
    if os.path.exists(checkpoint_path):
        model.load_state_dict(model_state_dict)

    model = model.to(device)
    intent_loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_data_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    while epoch < args.epochs:
        print(f'Epoch {epoch + 1}/{args.epochs}')
        print('-' * 10)

        train_model(
            model,
            train_data_loader,
            intent_loss_fn,
            optimizer,
            device,
            scheduler
        )

        epoch += 1

    # save predictions to CSV
    predict(
        model,
        test_data_loader,
        intent_loss_fn,
        device,
        args.out_dir
    )

    # save state dictionary
    last_checkpoint_state_dict = {
        'epoch': epoch + 1,  # epoch to continue training from
        'args': args,
        'model_state_dict': model.state_dict()
    }
    torch.save(last_checkpoint_state_dict, checkpoint_path)


if __name__ == '__main__':
    main()