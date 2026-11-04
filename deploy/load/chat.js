// k6 alternative to the Locust file. Run: k6 run deploy/load/chat.js
import http from "k6/http";
import { check } from "k6";

export const options = {
  vus: 10,
  duration: "20s",
};

export default function () {
  const response = http.post(
    "http://127.0.0.1:8000/v1/chat",
    JSON.stringify({
      message: "What is the status of order ORD-1001?",
      customer_id: "CUST-1001",
    }),
    {
      headers: {
        Authorization: "Bearer dev-token",
        "Content-Type": "application/json",
      },
    },
  );
  check(response, { "status is 200": (res) => res.status === 200 });
}
