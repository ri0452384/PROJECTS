#include <stdio.h>
#include <stdlib.h>
#include <windows.h>



//executes a non-built-in command
int shell_execute(int argc, TCHAR *argv[])
{
    printf("Command: %s\n", argv[0]);
    printf("Command: %s\n", argv[1]);
    return 1;
    /*
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory( &si, sizeof(si) );
    si.cb = sizeof(si);
    ZeroMemory( &pi, sizeof(pi) );

    if( argc != 2 )
    {
        printf("Usage: %s [cmdline]\n", argv[0]);
        return;
    }

    // Start the child process.
    if( !CreateProcess( NULL,   // No module name (use command line)
        argv[1],        // Command line
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        0,              // No creation flags
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory
        &si,            // Pointer to STARTUPINFO structure
        &pi )           // Pointer to PROCESS_INFORMATION structure
    )
    {
        printf( "CreateProcess failed (%d).\n", GetLastError() );
        return;
    }

    // Wait until child process exits.
    WaitForSingleObject( pi.hProcess, INFINITE );

    // Close process and thread handles.
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );

    return 1;
    */
}


#define TOKEN_BUFFER_SIZE 64
#define TOKEN_DELIMITERS " \t\r\n\a"
//should return a NULL-terminated array of pointers
char **shell_tokenize(char *line)
{
  int buffer_size = TOKEN_BUFFER_SIZE, position = 0;
  char **tokens = malloc(buffer_size * sizeof(char*));
  char *token;

  if (!tokens) {
    fprintf(stderr, "Shell: Unable to allocate!\n");
    exit(EXIT_FAILURE);
  }

  token = strtok(line, TOKEN_DELIMITERS);
  while (token != NULL) {
    tokens[position] = token;
    position++;

    if (position >= buffer_size) {
      buffer_size += TOKEN_BUFFER_SIZE;
      tokens = realloc(tokens, buffer_size * sizeof(char*));
      if (!tokens) {
        fprintf(stderr, "Shell: Unable to re-allocate!\n");
        exit(EXIT_FAILURE);
      }
    }

    token = strtok(NULL, TOKEN_DELIMITERS);
  }
  tokens[position] = NULL;
  return tokens;
}



//used to get a line from the user
#define LINE_BUFFER_SIZE 1024
char *shell_read_line(void)
{
  int buffer_size = LINE_BUFFER_SIZE;
  int position = 0;
  char *buffer = malloc(sizeof(char) * buffer_size);
  int c;

  if (!buffer) {
    fprintf(stderr, "Unable to allocate!\n");
    exit(EXIT_FAILURE);
  }

  while (1) {
    // Read a character
    c = getchar();

    // should terminate with a NULL character
    if (c == EOF || c == '\n') {
      buffer[position] = '\0';
      return buffer;
    } else {
      buffer[position] = c;
    }
    position++;

    // allocate more space if it exceeds the buffer
    if (position >= buffer_size) {
      buffer_size += LINE_BUFFER_SIZE;
      buffer = realloc(buffer, buffer_size);
      if (!buffer) {
        fprintf(stderr, "Unable to allocate!\n");
        exit(EXIT_FAILURE);
      }
    }
  }
}

#define BUFSIZE MAX_PATH
TCHAR Buffer[BUFSIZE+1];
//main loop, used to receive user input over and over until exit is called.
void shell_loop(void)
{

  char *user_input;
  char **arguments;
  int keep_running;
  DWORD dwRet;

    dwRet = GetCurrentDirectory(BUFSIZE, Buffer);
  do {
    printf(("%s> "), Buffer);
    //printf("%p> ", &dwRet);
    user_input = shell_read_line();
    arguments = shell_tokenize(user_input);
    keep_running = shell_execute(user_input,arguments);

    free(user_input);
    free(arguments);
  } while (keep_running);
}

//header function, nothing but loop and exiting the program.
int main(void)
{

    shell_loop();
    return EXIT_SUCCESS;
}
