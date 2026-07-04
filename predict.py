import joblib

# Load trained model
model = joblib.load("models/titanic_model.pkl")

print("===== Titanic Survival Prediction =====")

PassengerId = int(input("Passenger ID: "))
Pclass = int(input("Passenger Class (1/2/3): "))
Sex = int(input("Sex (0 = Male, 1 = Female): "))
SibSp = int(input("Siblings/Spouse: "))
Parch = int(input("Parents/Children: "))
Fare = float(input("Fare: "))
Embarked = int(input("Embarked (0=S, 1=C, 2=Q): "))
AgeGroup = int(input("Age Group (0=Child, 1=Teen, 2=Adult, 3=Senior): "))
FamilySize = int(input("Family Size: "))
IsAlone = int(input("Is Alone? (0=No, 1=Yes): "))
Title = int(input("Title (0=Mr, 1=Miss, 2=Mrs, 3=Master): "))

sample = [[PassengerId, Pclass, Sex, SibSp, Parch, Fare,
           Embarked, AgeGroup, FamilySize, IsAlone, Title]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\n✅ Passenger Survived")
else:
    print("\n❌ Passenger Did Not Survive")