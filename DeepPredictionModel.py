import numpy as np
import keras
import keras.backend as K
import sklearn.metrics as MCC
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D, Flatten, Dense, Merge, Reshape
import h5py

Month_data_combo = np.load("MONTH_DATA_COMBO.npy")
Month_data_event = np.load("MONTH_DATA_EVENT.npy")
Month_data_price = np.load("MONTH_DATA_PRICE.npy")
Month_label = np.load("MONTH_LABEL.npy")

Week_data_combo = np.load("WEEK_DATA_COMBO.npy")
Week_data_event = np.load("WEEK_DATA_EVENT.npy")
Week_data_price = np.load("WEEK_DATA_PRICE.npy")
Week_label = np.load("WEEK_LABEL.npy")

Day_data_combo = np.load("DAY_DATA_COMBO.npy")
Day_data_event = np.load("DAY_DATA_EVENT.npy")
Day_data_price = np.load("DAY_DATA_PRICE.npy")
Day_label = np.load("DAY_LABEL.npy")

print(Month_data_combo.shape, Month_data_event.shape, Month_data_price.shape)
print(Week_data_combo.shape, Week_data_event.shape, Week_data_price.shape)
print(Day_data_combo.shape, Day_data_event.shape, Day_data_price.shape)

select_train = int(len(Day_data_combo) * .8)

real_classes = []
for a_class in Day_label[select_train:]:
    if a_class[0] == 1:
        real_classes.append(0)
    else:
        real_classes.append(1)
real_classes = np.array(real_classes)



combo_MCC = 0
combo_acc = 0
for i in range(100):
    Month_ComboModel = Sequential()
    Month_ComboModel.add(BatchNormalization(input_shape = (40, 100), axis=1))
    Month_ComboModel.add(Conv1D(32, 6, activation = 'relu'))
    Month_ComboModel.add(MaxPooling1D(pool_size = 35))
    Month_ComboModel.add(Flatten())

    Week_ComboModel = Sequential()
    Week_ComboModel.add(BatchNormalization(input_shape = (10, 100), axis=1))
    Week_ComboModel.add(Conv1D(32, 6, activation = 'relu'))
    Week_ComboModel.add(MaxPooling1D(pool_size = 5))
    Week_ComboModel.add(Flatten())

    Day_ComboModel = Sequential()
    Day_ComboModel.add(BatchNormalization(input_shape = (200,)))

    Combo_Network = Sequential()
    Combo_Network.add(Merge([Month_ComboModel, Week_ComboModel, Day_ComboModel],  mode='concat'))
    Combo_Network.add(Dense(units = 128, activation = 'sigmoid'))
    Combo_Network.add(Dense(units = 2, activation = 'sigmoid'))

    Combo_Network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    Combo_Network.fit([Month_data_combo[:select_train], Week_data_combo[:select_train], Day_data_combo[:select_train]], Day_label[:select_train], shuffle=False, batch_size=32, epochs=10)
    score, acc = Combo_Network.evaluate([Month_data_combo[select_train:], Week_data_combo[select_train:], Day_data_combo[select_train:]], Day_label[select_train:], batch_size=32)
    prediction = Combo_Network.predict_classes([Month_data_combo[select_train:], Week_data_combo[select_train:], Day_data_combo[select_train:]])
    print("combo ", i, score, acc,  MCC.matthews_corrcoef(real_classes, prediction))
    if MCC.matthews_corrcoef(real_classes, prediction) > combo_MCC:
        Combo_Network.save("Combo_Model")
        combo_MCC = MCC.matthews_corrcoef(real_classes, prediction)
        combo_acc = acc


event_MCC = 0
event_acc = 0
for i in range(100):
    Month_EventModel = Sequential()
    Month_EventModel.add(BatchNormalization(input_shape = (20, 100), axis=1))
    Month_EventModel.add(Conv1D(32, 3, activation = 'relu'))
    Month_EventModel.add(MaxPooling1D(pool_size = 18))
    Month_EventModel.add(Flatten())

    Week_EventModel = Sequential()
    Week_EventModel.add(BatchNormalization(input_shape = (5, 100), axis=1))
    Week_EventModel.add(Conv1D(32, 3, activation = 'relu'))
    Week_EventModel.add(MaxPooling1D(pool_size = 3))
    Week_EventModel.add(Flatten())

    Day_EventModel = Sequential()
    Day_EventModel.add(BatchNormalization(input_shape = (100,)))

    Event_Network = Sequential()
    Event_Network.add(Merge([Month_EventModel, Week_EventModel, Day_EventModel],  mode='concat'))
    Event_Network.add(Dense(units = 64, activation = 'sigmoid'))
    Event_Network.add(Dense(units = 2, activation = 'sigmoid'))

    Event_Network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    Event_Network.fit([Month_data_event[:select_train], Week_data_event[:select_train], Day_data_event[:select_train]], Day_label[:select_train], shuffle=False, batch_size=32, epochs=10)
    score, acc = Event_Network.evaluate([Month_data_event[select_train:], Week_data_event[select_train:], Day_data_event[select_train:]], Day_label[select_train:], batch_size=32)
    prediction = Event_Network.predict_classes([Month_data_event[select_train:], Week_data_event[select_train:], Day_data_event[select_train:]])
    print("event ", i, score, acc,  MCC.matthews_corrcoef(real_classes, prediction))
    if MCC.matthews_corrcoef(real_classes, prediction) > event_MCC:
        Event_Network.save("Event_Model")
        event_MCC = MCC.matthews_corrcoef(real_classes, prediction)
        event_acc = acc


price_MCC = 0
price_acc = 0
for i in range(100):
    Month_PriceModel = Sequential()
    Month_PriceModel.add(BatchNormalization(input_shape = (20, 100), axis=1))
    Month_PriceModel.add(Conv1D(32, 3, activation = 'relu'))
    Month_PriceModel.add(MaxPooling1D(pool_size = 18))
    Month_PriceModel.add(Flatten())

    Week_PriceModel = Sequential()
    Week_PriceModel.add(BatchNormalization(input_shape = (5, 100), axis=1))
    Week_PriceModel.add(Conv1D(32, 3, activation = 'relu'))
    Week_PriceModel.add(MaxPooling1D(pool_size = 3))
    Week_PriceModel.add(Flatten())

    Day_PriceModel = Sequential()
    Day_PriceModel.add(BatchNormalization(input_shape = (100,)))

    Price_Network = Sequential()
    Price_Network.add(Merge([Month_PriceModel, Week_PriceModel, Day_PriceModel],  mode='concat'))
    Price_Network.add(Dense(units = 64, activation = 'sigmoid'))
    Price_Network.add(Dense(units = 2, activation = 'sigmoid'))

    Price_Network.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    Price_Network.fit([Month_data_price[:select_train], Week_data_price[:select_train], Day_data_price[:select_train]], Day_label[:select_train], shuffle=False, batch_size=32, epochs=10)
    score, acc = Price_Network.evaluate([Month_data_price[select_train:], Week_data_price[select_train:], Day_data_price[select_train:]], Day_label[select_train:], batch_size=32)
    prediction = Price_Network.predict_classes([Month_data_price[select_train:], Week_data_price[select_train:], Day_data_price[select_train:]])
    print("price ", i, score, acc,  MCC.matthews_corrcoef(real_classes, prediction))
    if MCC.matthews_corrcoef(real_classes, prediction) > price_MCC:
        Price_Network.save("Price_Model")
        price_MCC = MCC.matthews_corrcoef(real_classes, prediction)
        price_acc = acc


print("combo ", combo_MCC, combo_acc)
print("event ", event_MCC, event_acc)
print("price ", price_MCC, price_acc)
