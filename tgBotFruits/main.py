import telebot;
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TOKEN = "5674283096:AAGGvFRPH-2dCavJ8QYCjxxTFdT_J6oTx0w"

bot = telebot.TeleBot(TOKEN)

model= keras.models.load_model('../Fruits_v04.h5')

train_class = ['Яблоко', 'Банан', 'Капуста', 'Вишня', 'Драконий фрукт', 'Манго', 'Апельсин', 'Папайя',  'Ананас']


def predict(img_name, model):
    img = image.load_img(img_name,target_size=(224,224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    prediction = model.predict(img.reshape(1,224,224,3))
    output = np.argmax(prediction)
    return train_class[output]

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Привет, я могу определить овощь и фрукт по фото")
    bot.send_message(message.chat.id, "Вот список овощей и фруктов которые я распознаю: \n\t\t\tЯблоко, \n\t\t\tБанан, "
                                      "\n\t\t\tКапуста,"
                                      "\n\t\t\tВишня, \n\t\t\tДраконий фрукт, \n\t\t\tМанго, \n\t\t\tАпельсин, \n\t\t\tПапайя, \n\t\t\tАнанас")


@bot.message_handler(content_types=['photo'])

def handle_docs_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)

        downloaded_file = bot.download_file(file_info.file_path)

        src = 'tmp/' + file_info.file_path

        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)

        bot.reply_to(message, "Фото обрабатывается")

        bot.reply_to(message, predict(src, model))

        os.remove(src)

    except Exception as e:
        bot.reply_to(message, e)


if __name__ == '__main__':
    bot.polling(none_stop=True)