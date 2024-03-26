#NAME : santhosh Arunagiri
#student ID : 201586816


import random
import datetime

#creating a class Basic account
class BasicAccount:
    numAccounts = 0
#initializer
    def __init__(self, acName: str, openingBalance: float):
        BasicAccount.numAccounts += 1
        self.name = acName
        self.acNum = self.generateAcNum()
        self.balance = openingBalance
        self.cardNum = self.generateCardNum()
        self.cardExp = self.generateCardExp()

    def __str__(self):
        return f"Account name: {self.name}\nBalance: {self.balance}"

#generating account number

    def generateAcNum(self):
        # Generate a unique account number
        return self.numAccounts

#generating Card number
    def generateCardNum(self):
        # Generate a unique card number
        return random.randint(1000000000000000, 9999999999999999)

#generating expiry date for card in format (mm/yy)
    def generateCardExp(self):
        today = datetime.date.today()
        year = int(str(today.year + 3)[-2:])
        month = int(today.month)
        return (month,year)

#define method deposit( to deposit amount into account) 
    def deposit(self, amount: float):
        if amount <= 0:
            print('Invalid deposit amount')
            return

        self.balance += amount

#define method withdraw (to withdraw amount from account)
    def withdraw(self, amount : float):
        if amount <= 0:
            print('Can not withdraw £',str(amount))
        if amount > self.balance:
            print('Can not withdraw £',str(amount))
            return False
        else:
            self.balance -= amount
            print(self.name,'has withdrawn £',str(amount), 'New balance is £', str(self.balance))

#define method getavailableBalance (this returns the standing balance of the account)
    def getAvailableBalance(self):
        return self.balance

#define method getBalance (this returns the balance of the account)
    def getBalance(self):
        return self.balance

#define method printBalance (this prints the standing balance of the account )        
    def printBalance(self):
        print(f"Current balance: {self.balance}")

#define method getName ( this returns the name of the account holder)
    def getName(self):
        return self.name

#define method getAcNum (this returns the account number)
    def getAcNum(self):
        return str(self.acNum)

#define method issueNewCard (this creates a new card number and Expiry for the card)
    def issueNewCard(self):
        self.cardNum = self.generateCardNum()
        self.cardExp = self.generateCardExp()

#define method closeAccount (this closes the account after returning the balance amount to the account holder)
    def closeAccount(self):
        self.name = ""
        self.acNum = 0
        self.balance = 0
        self.cardNum = ""
        self.cardExp = (0, 0)
        self.withdraw(self.balance)
        return True

#chreating a class Premium account
class PremiumAccount(BasicAccount):
    def __init__(self, acName: str, openingBalance: float, initialOverdraft: float):
        super().__init__(acName, openingBalance)
        self.overdraft = True
        self.overdraftLimit = initialOverdraft


    def __str__(self):
        return super().__str__() + f"\nOverdraft limit: {self.overdraftLimit}"

#define method setoverdraftlimt (it sets the overdraft limit for the account)
    def setOverdraftLimit(self, newLimit: float):
        self.overdraftLimit = newLimit

#define method getAvailableBalance (it returns the available balance including the overdraft)
    def getAvailableBalance(self):
        return self.balance + self.overdraftLimit

#define method withdraw (it returns and prints the withdrawn amount from account including the overdraft)
    def withdraw(self, amount: float):
        if amount <= 0:
            print('Can not withdraw £',str(amount))
            return False
        if amount > self.balance + self.overdraftLimit:
            print('Can not withdraw £',str(amount))
            return False
        else:
            self.balance -= amount

            print(self.name,'has withdrawn £', str(amount), 'New balance is £', str(self.balance))
        return True

#define method printbalance (it prints the balance of the account after the overdraft amount if applicable)
    def printBalance(self):
        self.overdraftLimit += self.balance
        print(f"Current balance: {self.balance}\nOverdraft limit: {self.overdraftLimit}")

#define method close account (it returns the account close if the balance is zero, or withdraws the amount and closes the account if the balance is positive)
#if the balance is negative it doesnt allow to close the account
    def closeAccount(self):
        if self.overdraft > self.balance:
            print('Can not close account due to customer being overdrawn by £', self.overdraft)
            return False
        if self.overdraft < self.balance:
            super().closeAccount()
            self.withdraw(self.balance - self.overdraft)
            return True