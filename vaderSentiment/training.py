import pandas as pd
import wandb
import os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch import nn
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

warriner = pd.read_csv('C:/Users/hplis/OneDrive/Desktop/Koła/open_ai/Ratings_Warriner_et_al.csv')
bradley = pd.read_csv('https://raw.githubusercontent.com/mileszim/anew_formats/master/csv/all.csv')

model1 = "finiteautomata/bertweet-base-emotion-analysis"
model2 = "nghuyong/ernie-2.0-en"

# delete nan in "Words"
warriner = warriner.dropna(subset=['Word'])

warriner['norm_valence'] = (warriner['V.Mean.Sum'] - min(warriner['V.Mean.Sum'])) / (max(warriner['V.Mean.Sum']) - min(warriner['V.Mean.Sum']))





np.random.seed(112)

common = [x for x in list(warriner.Word.values) if x in list(bradley.Description.values)]
df_test = warriner[warriner.Word.isin(common)]
warriner = warriner[~warriner.Word.isin(common)]

df_train = warriner.sample(frac=0.9, random_state=42)
df_val = warriner.drop(df_train.index)


vader = pd.read_csv('vaderSentiment/vader_lexicon.txt', delimiter='\t')
columns = ['sign', 'negative', 'positive', 'neutral']
vader.columns = columns
vader['norm_valence'] = (vader['negative'] - min(vader['negative'])) / (max(vader['negative']) - min(vader['negative']))
vader = vader[~vader.sign.isin(list(warriner.Word.values))]
vader['Word'] = vader['sign']
# append vader to df_train
a = df_train.append(vader)


# save
# df_train.to_csv('vader_warriner_train.csv', index=False)
# df_test.to_csv('warriner_anew_test.csv', index=False)
# df_val.to_csv('warriner_anew_val.csv', index=False)

# append AoA and Concreteness scores to the previously generated train/val/test sets
df_train = pd.read_csv('vader_warriner_train.csv')


df_test = pd.read_csv('warriner_anew_test.csv')



df_val = pd.read_csv('warriner_anew_val.csv')



tokenizer2 = AutoTokenizer.from_pretrained(model1)

tokenizer3 = AutoTokenizer.from_pretrained(model2)

# Valence_M
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels_valence = df['norm_valence'].values.astype(float)

        self.texts2 = [tokenizer2(str(text).lower(),
                               padding='max_length', max_length = 8, truncation=True,
                                return_tensors="pt") for text in df['Word']]

        self.texts3 = [tokenizer3(str(text).lower(),
                               padding='max_length', max_length = 8, truncation=True,
                                return_tensors="pt") for text in df['Word']]

    def classes(self):
        return self.labels_valence

    def __len__(self):
        return len(self.labels_valence)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels_valence[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts2[idx], self.texts3[idx]

    def __getitem__(self, idx):

        batch_texts2, batch_texts3 = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)


        return batch_texts2, batch_texts3, batch_y


class BertRegression(nn.Module):

    def __init__(self, dropout=0.1, hidden_dim_valence=1536):

        super(BertRegression, self).__init__()

        self.bert1 = AutoModel.from_pretrained(model1)
        self.bert2 = AutoModel.from_pretrained(model2)

        self.valence = nn.Linear(hidden_dim_valence, 1)

        self.l_1_valence = nn.Linear(hidden_dim_valence, hidden_dim_valence)


        self.layer_norm = nn.LayerNorm(hidden_dim_valence)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id1, mask1, input_id3, mask3):
        _, y = self.bert1(input_ids = input_id1, attention_mask=mask1, return_dict=False)
        _, z = self.bert2(input_ids = input_id3, attention_mask=mask3, return_dict=False)
        x = torch.cat((y, z), dim=1)
        x = self.dropout(x)


        valence_all = self.dropout(self.relu(self.layer_norm(self.l_1_valence(x) + x)))
        valence = self.sigmoid(self.valence(valence_all))



        return valence




epochs = 1000
model = BertRegression()


train, val = Dataset(df_train), Dataset(df_val)

batch_size = 100
train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

criterion1 = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=5e-5,
                  eps=1e-8,  # Epsilon
                  weight_decay=0.3,
                  amsgrad=True,
                  betas = (0.9, 0.999))

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=600,
                                            num_training_steps=len(train_dataloader) * epochs)


if use_cuda:
    model = model.cuda()
    criterion1 = criterion1.cuda()




wandb.init(project="affect_anew", entity="hubertp")
wandb.watch(model, log_freq=5)


h = 0
best_loss = 150
best_corr_total = 0
for epoch_num in range(epochs):
    total_loss_train = 0

    for train_input1, train_input2, (valence) in tqdm(train_dataloader):
        mask1 = train_input1['attention_mask'].to(device)
        input_id1 = train_input1['input_ids'].squeeze(1).to(device)
        mask2 = train_input2['attention_mask'].to(device)
        input_id2 = train_input2['input_ids'].squeeze(1).to(device)
        valence = valence.to(device)
        output1 = model(input_id1, mask1, input_id2, mask2)

        del input_id1, mask1, input_id2, mask2


        l1 = criterion1(output1.float(), valence.view(-1,1).float())

        batch_loss = l1

        total_loss_train += batch_loss.item()
        model.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

    total_acc_val = 0
    total_loss_val = 0
    total_corr_valence = 0
    total_corr = 0
    with torch.no_grad():
        for val_input1, val_input2, (val_valence) in val_dataloader:

            mask1 = val_input1['attention_mask'].to(device)
            input_id1 = val_input1['input_ids'].squeeze(1).to(device)
            mask2 = val_input2['attention_mask'].to(device)
            input_id2 = val_input2['input_ids'].squeeze(1).to(device)
            val_valence = val_valence.to(device)
            output1 = model(input_id1, mask1, input_id2, mask2)

            l1 = criterion1(output1.float(), val_valence.view(-1,1).float())


            batch_loss = l1
            total_loss_val += batch_loss.item()

            output1 = output1.cpu().detach().view(-1).numpy()
            val_valence = val_valence.cpu().detach().numpy()


            total_corr_valence += np.corrcoef(output1, val_valence)[0, 1]

            total_corr = total_corr_valence


        if best_corr_total < total_corr / len(val_dataloader):
            best_corr_total = total_corr / len(val_dataloader)
            torch.save(model.state_dict(), 'models/english.pth')


    if epoch_num % 2 == 0:
        wandb.log({"loss": total_loss_train / len(df_train), "lr": scheduler.get_last_lr()[0], "epoch": epoch_num, "val_loss": total_loss_val/ len(df_val), "val_corr_valence": total_corr_valence / len(val_dataloader)})
    print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(df_train): .10f} \
            | Val Loss: {total_loss_val / len(df_val): .10f} | corr_valence: {total_corr_valence / len(val_dataloader): .10f}')



########################################################################################################################
########################################################################################################################
########################################################################################################################

model.load_state_dict(torch.load('models/english.pth'))

# from torch.utils.tensorboard import SummaryWriter
#
# # default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')
#
# writer.add_graph(model, (mask1[0,:,:], input_id1, mask2[0,:,:], input_id2))


# torch.save(model, 'models/tryout.pth')


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    loss = 0
    with torch.no_grad():
        preval, trueval = [], []

        for test_input2, test_input3, (test_valence) in test_dataloader:
            mask2 = test_input2['attention_mask'].to(device)
            input_id2 = test_input2['input_ids'].squeeze(1).to(device)
            mask3 = test_input3['attention_mask'].to(device)
            input_id3 = test_input3['input_ids'].squeeze(1).to(device)
            test_valence = test_valence.to(device)

            output1  = model(input_id2, mask2, input_id3, mask3)
            # batch_loss = criterion(output.float(), val_label.float())

            l1 = criterion1(output1.float(), test_valence.view(-1,1).float())
            loss += l1.item()
            preval.extend([p for p in output1.cpu()])

            trueval.extend([t for t in test_valence.cpu()])

        print(f'Test Loss: {loss / len(test): .10f}')
            # print loss
    return preval, trueval

pred_val, true_val= evaluate(model, df_test)


diffs = []
for i in range(len(pred_val)):
    diffs.append(float(abs(pred_val[i] - true_val[i])))

mean_val = sum(diffs) / len(diffs)
# compute correlation
p_v = [float(v) for v in pred_val]
t_v = [float(v) for v in true_val]
corr_val = np.corrcoef(p_v, t_v)[0, 1]


print(corr_val)

