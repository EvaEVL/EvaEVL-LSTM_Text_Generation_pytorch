from aiogram import Bot, Dispatcher, executor, types
from modelLSTM import Model
from modeluser import Trainer
from data import Data
import torch
import random

API_TOKEN = ""
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
TRAIN_MODEL = False

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Hello!")


@dp.message_handler()
async def echo(
        message: types.Message):
    await message.answer(t.evaluate(start_seq=message.text + ' ', prediction_len=random.randint(50, 400)))


if __name__ == '__main__':
    args = {
        'path': 'data_corpus.csv',
        'ebm_size': 128,
        'hidden_size': 128
    }
    d = Data(args)  # создаем объект класс Data, чтобы было легче работать с предложениями
    corpus, tokens_to_id, idx_to_tokens = d.read_file()
    args['input_size'] = len(tokens_to_id)
    model = Model(input_size=args['input_size'], hidden_size=args['hidden_size'], embedding_size=args['ebm_size'],
                  n_layer=2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    t = Trainer(model, 32, args, d, tokens_to_id, idx_to_tokens, num_epoch=1, eval=True)

    model.load_state_dict(torch.load('model_path', map_location=torch.device('cpu')))
    # t.train(num_epoch=100, eval=True)
    executor.start_polling(dp, skip_updates=True)
