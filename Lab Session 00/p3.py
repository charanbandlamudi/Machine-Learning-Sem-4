# Questions 5 to 8
s1 = "I am a great learner. I am going to have an awesome life."
c = s1.count("am")  # Count occurrences of "am"
print("Number of occurrences of 'am' in s1: ", c)

s2 = "I work hard and shall be rewarded well"
s3 = s1 + " " + s2
print("Combined string (s3):")
print(s3)

# Split s3 by whitespaces
a1 = s3.split()

# Filter out words "I", "am", "and", "to" safely
filtered_a1 = [word for word in a1 if word not in ["I", "am", "and", "to"]]

print("Array without 'I', 'am', 'and', 'to':")
print(filtered_a1)


