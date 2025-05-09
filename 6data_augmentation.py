import pandas as pd
import numpy as np

#Load
data = pd.read_csv('processed_data.csv')

np.random.seed(42)

#generate synthetic data
def augment_data(data, num_samples, noise_scale=0.1):
    #identifing numerical and categorial columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    
    augmented_data = []
    for _ in range(num_samples):
         #randomly sample original data
        sample = data.sample(n=1)
        
        #Generate new numerical data
        new_numerical_values = sample[numerical_cols].values[0] + np.random.normal(0, noise_scale, size=sample[numerical_cols].shape)
        
        #new DataFrame
        new_sample_df = pd.DataFrame(new_numerical_values.reshape(1, -1), columns=numerical_cols)
        
        #categorical features
        for col in categorical_cols:
            new_sample_df[col] = sample[col].values[0]
        
        augmented_data.append(new_sample_df)

    return pd.concat(augmented_data, ignore_index=True)

# Augment dataset
augmented_data = augment_data(data, num_samples=500)  # 500 new samples

#Combine original and augmented data
combined_data = pd.concat([data, augmented_data], ignore_index=True)

#Save
combined_data.to_csv('augmented_data.csv', index=False)

print("Data augmentation complete. Augmented data saved to 'augmented_data.csv'.")
print(f"Original data shape: {data.shape}")
print(f"Augmented data shape: {augmented_data.shape}")
print(f"Combined data shape: {combined_data.shape}")
