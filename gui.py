import tkinter as tk
from tkcalendar import Calendar, DateEntry
import datetime as dt
import tkinter.filedialog
from main import *


class mainFrame:
    def __init__(self):
        self.root=tk.Tk()
        self.root.minsize(500,500)
        self.root.geometry("500x500")

        self.loadedModel=None

        self.startTrainingLabel=tk.Label(text="Wybierz datę rozpoczęcia dla treningu")
        self.startTrainingLabel.pack(padx=10, pady=5)
        self.startTrainingDateEntry = DateEntry(self.root, width=12, background='darkblue',
                        foreground='white', borderwidth=2)
        self.startTrainingDateEntry.pack(padx=10, pady=5)

        self.endTrainingLabel = tk.Label(text="Wybierz datę zakończenia dla treningu")
        self.endTrainingLabel.pack(padx=10, pady=5)
        self.endTrainingDateEntry = DateEntry(self.root, width=12, background='darkblue',
                                   foreground='white', borderwidth=2)
        self.endTrainingDateEntry.pack(padx=10, pady=5)


        #self.startTestLabel = tk.Label(text="Wybierz datę rozpoczęcia dla testów")
        #self.startTestLabel.pack(padx=10, pady=5)
        #self.startTestingDateEntry = DateEntry(self.root, width=12, background='darkblue',
                                           #foreground='white', borderwidth=2)
        #self.startTestingDateEntry.pack(padx=10, pady=5)

        self.endTestLabel = tk.Label(text="Wybierz datę zakończenia dla testów")
        self.endTestLabel.pack(padx=10, pady=5)
        self.endTestingDateEntry = DateEntry(self.root, width=12, background='darkblue',
                                         foreground='white', borderwidth=2)
        self.endTestingDateEntry.pack(padx=10, pady=5)



        self.tickerIndexLabel=tk.Label(text="Wprowadź indeks firmy")
        self.tickerIndexLabel.pack(padx=10,pady=5)

        self.tickerIndexEntry=tk.Entry(self.root)
        self.tickerIndexEntry.pack(padx=10, pady=5)

        self.epochNumberLabel = tk.Label(text="Wprowadź liczbe epok")
        self.epochNumberLabel.pack(padx=10, pady=5)

        self.epochNumberEntry=tk.Entry(self.root)
        self.epochNumberEntry.pack(padx=10, pady=5)


        self.savemodelVariable=tk.IntVar()

        self.saveModelCheckbox=tk.Checkbutton(text="zapisz model",variable=self.savemodelVariable)
        self.saveModelCheckbox.pack(padx=10,pady=5)

        self.loadModelVariable=tk.IntVar()

        self.loadModelButton=tk.Checkbutton(text="wczytaj model",variable=self.loadModelVariable)
        self.loadModelButton.pack(padx=10,pady=5)

        self.predDaysLabel = tk.Label(text="liczba dni wstecz do predykcji")
        self.predDaysLabel.pack(padx=10, pady=5)

        self.predDaysEntry=tk.Entry(self.root)
        self.predDaysEntry.pack(padx=10,pady=5)

        self.runButton=tk.Button(text="start",command=self.runProgram)
        self.runButton.pack(padx=10,pady=5)


        self.root.mainloop()
    def getDates(self):

        d1= dt.datetime.strptime(self.startTrainingDateEntry.get(),"%m/%d/%y")
        d2 = dt.datetime.strptime(self.endTrainingDateEntry.get(), "%m/%d/%y")
        #d3 = dt.datetime.strptime(self.startTestingDateEntry.get(), "%d/%m/%y")
        d3 = dt.datetime.strptime(self.endTestingDateEntry.get(), "%m/%d/%y")

        return d1,d2,d3

    #def loadModel(self):
        #filename= tkinter.filedialog.askopenfilename()
        #self.loadedModel=filename

    #def getLoadedModel(self):
        #return self.loadedModel
    def getIndexLlabel(self):
        return self.tickerIndexEntry.get()
    def getEpochNumber(self):
        return int(self.epochNumberEntry.get())
    def getSaveBool(self):
        if(self.savemodelVariable.get()==1):
            return True
        else:
            return False
    def getLoadModelBool(self):
        if(self.loadModelVariable.get()==1):
            return True
        else:
            return False
    def getPredDays(self):
        return int(self.predDaysEntry.get())
    def runProgram(self):
        d1,d2,d3=self.getDates()
        companyidx=self.getIndexLlabel()
        epochNumber=self.getEpochNumber()
        saveBool=self.getSaveBool()
        predDays=self.getPredDays()
        loadModelBool=self.getLoadModelBool()

        run(companyIndex=companyidx,date1=d1,date2=d2, date3=d3,
            predictDays=predDays,numberOfEpochs=epochNumber,
            saveBool=saveBool,loadModelBool=loadModelBool)


mainFrame()






