import axiosInstance from "../config/axiosInstance";

export const chatApi = async (question: string) => {
  const res = await axiosInstance.post("/chat", { question });
  return res.data;
};

