# MTS_Churn_Predict

Домашнее задание команды Dungeon machines в рамках школы МТС.Тета. 

# Краткий отчёт

Поставлена бизнес-цель уменьшение оттока клиентов, для достижения цели была решена задачу бинарной классификации. <br />
Был выбран датасет в сфере телеком. На основании этого датасета было проведёно предпроектное исследование. <br />
В датасете обнаружен дисбаланс классов, характерный для задачи оттока. Данные становятся достаточно хорошо разделимы нелинейной поверхностью. Были построены две baseline -  модели. <br />
При увеличении ROC-AUC на 10% эконом. эффект вырос на 275%.

Чтобы оценивать результат с течением времени, было принято решение, ввести новый признак Lifetime - время от начала заключения договора до даты выгрузки данных. Наш датасет был отсортирован по времени, построена базовая модель на более ранних данных. После этого были подобраны оптимальные параметры для случайного леса и бустинга, лучшее качество показал бустинг.<br />
По результатам разработки моделей было получено следующее качество: базовая модель random forest(roc-auc 0.881 на тесте, 0.912 на CV) и xgboost(roc-auc 0.901 на тесте, 0.928 на CV). Итоговый прирост составил 0.02 на тесте и 0.016 на CV. Мы наблюдаем несущественное изменение качества модели по фолдам относительно среднего значения. Модель ведет себя достаточно стабильно.

Наиболее важными признаками оказались роуминг и кол-во звонков в колл-центр. При подключенном роуминге и более трёх обращений в колл-центр доля оттока значительно возрастает. 
В подробном отчёте указаные сегменты, на которых модель показывает качество отличное от среднего.<br />
Для обучения модели были взяты данные с 281 дня по 105, для оценки деградаци со 105 по 1. За данный период тенденция падения качества модели не обнаружена.

Лучшая модель была использована при разработке демо-сервиса, который позволяет оценить работу построенной модели. На основе входных данных алгоритм выдаёт аналитику по оттоку в выбранных штатах. Сервис содержит инструкцию по применению.


# Источник данных

https://bigml.com/user/francisco/gallery/dataset/5163ad540c0b5e5b22000383

# Описание проекта

Вам предоставляется отчет и набор данных от телекоммуникационной компании. Данные содержат информацию о более чем трех тысячах пользователей, их демографических характеристиках, услугах, которыми они пользуются, длительности использования услуг оператора и сумме оплаты.

Задача - проанализировать данные и спрогнозировать отток пользователей (выявить людей, которые будут и не будут продлевать свой контракт). Удержание пользователей - одна из наиболее актуальных задач в областях, где распространение услуги составляет порядка 100%. Ярким примером такой области является сфера телекома. Удержание клиента обойдется компании дешевле, чем привлечение нового.

Прогнозируя отток, мы можем вовремя среагировать и постараться удержать клиента, который хочет уйти. На основании данных об услугах, которыми пользуется клиент, мы можем сделать ему специальное предложение, пытаясь изменить его решение покинуть оператора. Это сделает задачу удержания проще в реализации, чем задачу привлечения новых пользователей. На основе полученных данных команда DS проверяет возможно и эффективно ли строить и применять модель оттока. В положительном случае модель обучается, проводится A/B тест. Если результаты A/B теста экономически и статистически значимы, то принимается решение о внедрении модели в продакшн. Команде разработки даётся задание подготовить соответствующий функционал в приложении, а также автоматическую рассылку на почту и телефон. После этого выбирается периодичность с которой будет производится анализ на предмет оттока. В результате этого на серверах компании появляется соответствующая информация. Далее, в соотвествии с ожидаемым оттоком, клиенты получают информацию об акции/скидке и других предложениях. Позже комадна DS проверяет результаты A/B теста.

Подробное описание и реализация находится в telecom_churn.ipynb <br />










