import axios from 'axios';


const API_BASE_URL = 'https://age-i-kelly-huang.koyeb.app/';


const apiService = axios.create({
  baseURL: API_BASE_URL,
});

export const predictAge = async (image) => {
  try {
    const formData = new FormData();
    formData.append('file', image);

    const response = await apiService.post('/predict-age', formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
    });
    return response.data;
  } catch (err) {
    console.error('Error analyzing image:', err)
  }
};