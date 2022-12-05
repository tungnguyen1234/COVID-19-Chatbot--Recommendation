from data_processing import write_data, delete_data
from model import *

if __name__ == '__main__':
    main_path = "data/"
    writePath_chatbot = 'chatbot_raw.txt'
    writePath_covid = 'covid19_raw.txt'

    # process data
    # get the chatbot QA data
    addtional_data, sheet = 'COVID_Chatbot_data.xlsx', 'General QA'
    label = 0
    file2 = 'general_dataset.txt'
    final_chatbot = 'chatbot_related_dataset_final.txt'
    write_data(main_path, writePath_chatbot, addtional_data, sheet, label, file2, final_chatbot)

    # get the covid research QA data
    sheet = 'Covid data'
    label = 1
    file2 = 'covid_related_dataset.txt'
    final_covid = 'covid_related_dataset_final.txt'
    write_data(main_path, writePath_covid, addtional_data, sheet, label, file2, final_covid)

    # Check the overall performance of the model by LLO
    X,y = data_processing(main_path, final_chatbot, final_covid)
    
    # check the training and testing score of the model
    print("check the 1st pipeline model: SGD classifier")
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model_cv(pipeline1, X, y)
    model(pipeline1, X_train, X_test, y_train, y_test)
    
    # check the training and testing score of the model
    print("check the 2nd pipeline model: naive bayes")
    model_cv(pipeline2, X, y)
    model(pipeline2, X_train, X_test, y_train, y_test)
    
    # check the training and testing score of the model
    print("check the 3rd pipeline model: Losgistic regression")
    model_cv(pipeline3, X, y)
    model(pipeline3, X_train, X_test, y_train, y_test)

    # delete file to rewrite
    delete_data(main_path, writePath_chatbot)

    # delete file to rewrite
    delete_data(main_path, writePath_covid)

