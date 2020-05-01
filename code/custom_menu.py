import sys
import msvcrt
import traceback

def input_menu():
    sys.argv = []
    print("\nEnter the option number and hit enter to execute different recipe recommendation modules.")
    cmdinput = input("1. Run Recommendation Models. \n"
                     "2. Run Get Recipe Recommendation for Existing User.\n"
                     "3. New User.\n"
                     "4. Exit.\n")

    if cmdinput == '1':
        exec(open("custom_model.py").read())

    elif cmdinput == '2':
        userId = int(input("\nPlease Enter User Id: "))
        sys.argv = [userId]
        exec(open("custom_recommender.py").read())

    elif cmdinput == '3':
        userHt = int(input("\nPlease Enter User Height in (inches): "))
        userWt = int(input("Please Enter User Weigth in (lbs): "))
        userAge = int(input("Please Enter User age (> 18): "))
        userGender = str(input("Please Enter User gender as (male/female): "))
        print("Please Enter User activity type from below:")
        userAT = int(input("1. sedentary\n"
                         "2. lightly_active\n"
                         "3. moderately_active\n"
                         "4. very_active\n"
                         "5. extra_active\n"))

        if userAT > 5:
            print("Entered wrong option, exiting program...\n ")
            exit(0)

        #calcualte bmr
        height_mtr = userHt * 0.0254
        if userGender.lower() == 'male':
            bmr = 66 + (6.3 * userWt) + (12.9 * height_mtr) - (6.8 * userAge)
        else:
            bmr = 655 + (4.3 * userWt) + (4.7 * height_mtr) - (4.7 * userAge)

        #depending on user activity type, calculate cal per day and then cal per dish
        sedentary_mf = 1.2
        lightly_active_mf = 1.375
        moderately_active_mf = 1.55
        very_active_mf = 1.725
        extra_active_mf = 1.9
        cal_per_day = 0
        if userAT == 1: cal_per_day = bmr * sedentary_mf
        elif userAT == 2: cal_per_day = bmr * lightly_active_mf
        elif userAT == 3: cal_per_day = bmr * moderately_active_mf
        elif userAT == 4: cal_per_day = bmr * very_active_mf
        elif userAT == 5: cal_per_day = bmr * extra_active_mf
        cal_per_dish = cal_per_day / 3

        sys.argv = [cal_per_dish, True]
        exec(open("custom_recommender.py").read())

    elif cmdinput == '4':
        exit(0)

    else:
        print("Entered wrong option, exiting program...\n ")
        exit(0)

if __name__ == '__main__':
    while True:
        try:
            if msvcrt.kbhit() and msvcrt.getch()==chr(27).encode():
                print("\nEsc key pressed, exiting program...")
                sys.exit(0)
            else: input_menu()
        except:
            #print(sys.exc_info()[0])
            #print(traceback.format_exc())
            break
