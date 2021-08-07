// assign pin num
int right_pin = 12;
int left_pin = 13;
int forward_pin = 9;
int reverse_pin = 4;

// duration for output
int time = 1;
int time2 = 3;
// initial command
int command = 0;

void setup() {
 pinMode(right_pin, OUTPUT);
 pinMode(left_pin, OUTPUT);
 pinMode(forward_pin, OUTPUT);
 pinMode(reverse_pin, OUTPUT);
 Serial.begin(9600);
}

void loop() {
 //receive command
 if (Serial.available() > 0){
   command = Serial.read();
 }
 else{
   reset();
 }
  send_command(command,time);
}

void right(int time){
 digitalWrite(right_pin, LOW);
 delay(time);
 digitalWrite(right_pin, HIGH);
 delay(time2);
}

void left(int time){
 digitalWrite(left_pin, LOW);
 delay(time);
}

void forward(int time){
 digitalWrite(forward_pin, LOW);
 delay(time);
 digitalWrite(forward_pin, HIGH);
 delay(time2);
}

void reverse(int time){
 digitalWrite(reverse_pin, LOW);
 delay(time);
}

void forward_right(int time){
 digitalWrite(forward_pin, LOW);
 digitalWrite(right_pin, LOW);
 delay(time);
  digitalWrite(forward_pin, HIGH);
 digitalWrite(right_pin, HIGH);
 delay(time2);
}

void reverse_right(int time){
 digitalWrite(reverse_pin, LOW);
 digitalWrite(right_pin, LOW);
 delay(time);
}

void forward_left(int time){
 digitalWrite(forward_pin, LOW);
 digitalWrite(left_pin, LOW);
 delay(time);
  digitalWrite(forward_pin, HIGH);
 digitalWrite(left_pin, HIGH);
 delay(time2);
}

void reverse_left(int time){
 digitalWrite(reverse_pin, LOW);
 digitalWrite(left_pin, LOW);
 delay(time);
}

void reset(){
 digitalWrite(right_pin, HIGH);
 digitalWrite(left_pin, HIGH);
 digitalWrite(forward_pin, HIGH);
 digitalWrite(reverse_pin, HIGH);
}

void send_command(int command, int time){
 switch (command){

    //reset command
    case 0: reset(); break;

    // single command
    case 1: forward(time); break;
    case 2: reverse(time); break;
    case 3: right(time); break;
    case 4: left(time); break;

    //combination command
    case 6: forward_right(time); break;
    case 7: forward_left(time); break;
    case 8: reverse_right(time); break;
    case 9: reverse_left(time); break;

    default: Serial.print("Invalid Command\n");
   }
}