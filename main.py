import time

import telebot
import torch
import os
import shutil
import cv2
import moviepy.editor as moviepy

bot = telebot.TeleBot('5957713479:AAEEz7JOE1yHUdXx_1SNH7U5_q3U4DZvekk')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
URL = '/usr/share/app/'
mass = []


@bot.message_handler(commands=['start'])
def main(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}')


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = URL + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        send_photo(message, file_info, src)
    except Exception as e:
        bot.reply_to(message, e)


def send_photo(message, file_info, src):
    number = file_info.file_path[file_info.file_path.find('_') + 1:]
    number = number[:-4]
    model_file = ai_model_for_photo_processing(src, number)
    name = model_file.pandas().xyxy[0]
    send_src = URL + f'photos/file_{number}/file_{number}.jpg'
    bot.reply_to(message, f'Фотография обработана!')
    bot.send_photo(message.chat.id, open(send_src, 'rb'))
    # bot.send_text(message.chat.id, name)
    remove_photo(number, src)


def ai_model_for_photo_processing(src, number):
    model_file = model(src)
    model_file.save(labels=True, save_dir=URL + f'photos/file_{number}')
    return model_file


def remove_photo(number, src):
    os.remove(src)
    shutil.rmtree(URL + f'photos/file_{number}')


@bot.message_handler(content_types=['video'])
def get_video(message):
    try:
        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = URL + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        send_video(message, file_info, src)
    except Exception as e:
        bot.reply_to(message, e)


def send_video(message, file_info, src):
    number = file_info.file_path[file_info.file_path.find('_') + 1:]
    number = '.'.join(number.split('.')[:-1])
    ai_model_for_video_processing(src, number)
    send_src = URL + f'videos/model_videos/file_{number}.mp4'
    bot.reply_to(message, f'Видео обработано!')
    clip = moviepy.VideoFileClip(send_src)
    compressed = URL + f'videos/send/file_{number}.mp4'
    clip.write_videofile(compressed)
    bot.send_video(message.chat.id, open(compressed, 'rb'))
    remove_video(send_src, src, compressed)


def ai_model_for_video_processing(src, number):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_cod = cv2.VideoWriter_fourcc(*'XVID')
    video_output = cv2.VideoWriter(f'/usr/share/app/videos/model_videos/file_{number}.mp4', video_cod, fps,
                                   (frame_width,
                                    frame_height))
    shot = 1
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while shot < frame_number:
        ret, frame = cap.read()
        # assert ret

        frame = cv2.resize(frame, (frame_width, frame_height))
        results = score_frame(frame)
        frame = plot_boxes(results, frame)

        video_output.write(frame)
        shot += 1

    video_output.release()
    cv2.destroyAllWindows()


def score_frame(frame):
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


def remove_video(send_src, src, compressed):
    os.remove(src)
    os.remove(send_src)
    os.remove(compressed)


bot.polling(none_stop=True)
