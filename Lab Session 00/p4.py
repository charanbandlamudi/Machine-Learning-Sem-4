# Question-9
date="01-JUN-2021"
act_date=date.split("-")
print("Date is: ", act_date[0])
month_number={"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEPT":9,"OCT":10,"NOV":11,"DEC":12}
if act_date[1] in month_number:
    mth=month_number[act_date[1]]
print(f"Final Date: {act_date[0]}-{mth}-{act_date[2]}")