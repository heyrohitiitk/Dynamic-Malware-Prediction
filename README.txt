PROJECT TITLE : DYNAMIC MALWARE PREDICTION Using RNN
GROUP : TURING


DEPENDENCIES USED:
Python 2 and Python 3
Tensorflow==2.8.0
Scikit Learn==0.24.2
Pandas
Numpy
Psutil
BeautifulSoup
Selenium


FOLDERS AND FILES ATTACHED:
NOTE: Cuckoo setup needs to be done beforehand. Guest Machine should be windows7.
    1. data_extraction(folder)
        a. datacsv.py
        b. transfer.py
        c. virus.sh
    2. Demo
        a. Project_Demo.mp4
    3. Source Code
        I. experiments(folder)
            a. Configs.py
            b. Experiments.py
            c. RNN.py
            d. useful.py
        II. random_search_reults_x
            a. results.csv
        III. col_headers.py
        IV. labelling.ipynb
        V. main.py
        VI.means.npy
        VII. rnn.h5
        VIII. std.npy
        IX. tester.py
        X. testing.csv
        XI. wrapper.py 
    4. setup_files(folder)
        a. abstracts.py
        b. cuckoo.conf
        c. custom.py
    5. Makefile
    6. MalwarePrediction.pptx
    7. Project_Report.pdf
    7. README.txt
    


GUIDE:
	1. Cuckoo Configuration:
        I. Open the setup_files folder 
        II. Copy abstracts.py file to .cuckoo/analyzer/windows/lib/common folder
        III. Copy custom.py file to .cuckoo/analyzer/windows/modules/auxiliary folder
        IV. Copy cuckoo.conf to .cuckoo/conf folder
	2. Start the cuckoo server by typing "cuckoo" command in terminal.
	3. Copy files from data_extraction folder and paste into HOME directory.
    4. Data Extraction:
        I. The Malware files need to be tested, should present in HOME directory.
        II. Open the terminal and run command "sh virus.sh 'filename'".
        III. As a result, test_dataset.csv file will be generated.
    5. Go to main project folder
        I. Open the terminal here.
        II. Run command "make run".
        III. On the screen, select option to test the malware file.
        IV. Enter the path of test_dataset.csv file generated previously.
        V. Prediction result will be displayed on the screen.
