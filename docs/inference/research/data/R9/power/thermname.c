#include <stdio.h>
#include <notify.h>
#include <libkern/OSThermalNotification.h>
int main(void){
  int tok; uint64_t v=0;
  printf("kOSThermalNotificationPressureLevelName=%s\n", kOSThermalNotificationPressureLevelName);
  if (notify_register_check(kOSThermalNotificationPressureLevelName, &tok)==NOTIFY_STATUS_OK){ notify_get_state(tok,&v); printf("state=%llu\n",(unsigned long long)v);} 
  return 0;
}
