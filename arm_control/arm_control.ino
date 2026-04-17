#include <Servo.h>
#include <Wire.h>

Servo s[5];
int pins[5] = {A1, 2, 3, 12, 13};
int angles[5] = {90, 90, 90, 90, 90};

int16_t ax, ay, az, gx, gy, gz;
float pitch, roll;
float pitch0, roll0;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);
  
  Wire.beginTransmission(0x68);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
  
  for(int i = 0; i < 5; i++){
    s[i].attach(pins[i]);
    s[i].write(90);
  }
  delay(500);
  
  MPU6050_Read();
  pitch0 = pitch;
  roll0 = roll;
  
  Serial.println("OK");
}

void loop() {
  MPU6050_Read();
  
  float diffPitch = pitch - pitch0;
  float diffRoll = roll - roll0;
  
  int target1 = 90 + (int)(diffPitch * 2);
  int target2 = 90 + (int)(diffRoll * 2);
  int target3 = 90 + (int)(diffPitch * 1.5);
  int target4 = 90 + (int)(diffRoll * 1.5);
  int target5 = 90;
  
  target1 = constrain(target1, 0, 180);
  target2 = constrain(target2, 0, 180);
  target3 = constrain(target3, 0, 180);
  target4 = constrain(target4, 0, 180);
  
  s[0].write(target1);
  s[1].write(target2);
  s[2].write(target3);
  s[3].write(target4);
  s[4].write(target5);
  
  Serial.print("P:");
  Serial.print((int)pitch);
  Serial.print(" R:");
  Serial.print((int)roll);
  Serial.print(" -> ");
  Serial.print(target1);
  Serial.print(",");
  Serial.print(target2);
  Serial.print(",");
  Serial.print(target3);
  Serial.print(",");
  Serial.print(target4);
  Serial.print(",");
  Serial.println(target5);
  
  delay(10);
}

void MPU6050_Read() {
  Wire.beginTransmission(0x68);
  Wire.write(0x3B);
  Wire.requestFrom(0x68, 6);
  while(Wire.available() < 6);
  
  int16_t axh = (Wire.read() << 8) | Wire.read();
  int16_t ayh = (Wire.read() << 8) | Wire.read();
  int16_t azh = (Wire.read() << 8) | Wire.read();
  
  float accx = (float)axh / 32768.0 * 16.0;
  float accy = (float)ayh / 32768.0 * 16.0;
  float accz = (float)azh / 32768.0 * 16.0;
  
  pitch = atan2(accy, sqrt(accx * accx + accz * accz)) * 180.0 / PI;
  roll = atan2(-accx, accz) * 180.0 / PI;
}