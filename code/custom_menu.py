import sys
import traceback
import joblib
import os
from csv import writer

def input_menu():
    #sys.argv = []
    print("\nEnter the option number and hit enter to execute different recipe recommendation modules.")
    cmdinput = input("1. Create Recommendation Models. \n"
                     "2. Run Get Recipe Recommendation for Existing User.\n"
                     "3. New User.\n"
                     "4. Evaluate models.\n"
                     "5. Exit.\n")

    if cmdinput == '1' or cmdinput == '4':
        if cmdinput == '4': sys.argv = ['eval', True]
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

        recipe_name_to_be_rated = str(input("\nPlease enter Recipe Name that you like: "))
        rating_value = int(input("Please enter rating (1-5): "))
        import pandas as pd
        users_df = pd.read_csv(os.path.realpath('../data/clean/users.csv'))
        recipe_df = pd.read_csv(os.path.realpath('../data/clean/recipes.csv'))
        #get recipe id from name
        selected_recipe_id = recipe_df.loc[recipe_df['recipe_name'].str.lower() == recipe_name_to_be_rated.lower()]['recipe_id'].values
        #get max user id and add one to generate new (non colliding) user id
        new_user_id = max(users_df['user_id']) + 1
        #write user information to file
        with open(os.path.realpath('../data/clean/users.csv'), 'a+', newline='') as fobj:
            csv_writer = writer(fobj)
            csv_writer.writerow([new_user_id, userGender, userHt, userWt, userHt * 0.0254, userWt * 0.453592, 0.0, userAge, userAT, bmr, cal_per_day])
        fobj.close()
        # write ratings information to file
        with open(os.path.realpath('../data/clean/ratings.csv'), 'a+', newline='') as fobj:
            csv_writer = writer(fobj)
            csv_writer.writerow([new_user_id, selected_recipe_id[0], rating_value])
        fobj.close()
        print("New user information and ratings updated...")
        print("The new user id is: ", new_user_id)

    elif cmdinput == '5':
        print("Exiting program...\n ")
        sys.exit(0)

    else:
        print("Entered wrong option, exiting program...\n ")
        sys.exit(0)

if __name__ == '__main__':
    # try:
    #     input_menu()
    # except:
    #     print(sys.exc_info()[0])
    #     print(traceback.format_exc())
    # finally:
    #     input_menu()

    while True:
        try:
            input_menu()
        except:
            #print(sys.exc_info()[0])
            #print(traceback.format_exc())
            break
