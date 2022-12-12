#include "post_process.cpp"

#include <vector>
#include <array>
#include <queue>
#include <cmath>
#include <string>

template <class T>
using Vec1D = std::vector<T>;

template <class T>
using Vec2D = std::vector<Vec1D<T>>;

template <class T>
using Vec3D = std::vector<Vec2D<T>>;

gint frame_number = 0;

#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define CONFIG_PATH "deepstream_app_server_config.txt"
#define SIZE 256

gint ShutdownCommand = 0;
gint PoseWarning = 0;
gint PeopleWarning = 0;
gint NobodyWarning = 0;
gint SuspiciousItemWarning = 0;
gchar lockbuf[]="LOCK";

/*config vars*/  
gint port_lock = 0;
gint portnumber = 0;
gchar ipaddress[SIZE];
gint open_send_socket_count_limit = 0;
gint send_socket_count_limit = 0;
gint open_hand_lock = 0;

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
gint pose_estimation_muxer_output_width = 0;
gint pose_estimation_muxer_output_height = 0;

gint PoseWarningLimit = 15;
gint LeftArmMin = 45;
gint LeftArmMax = 145;
gint RightArmMin = 45;
gint RightArmMax = 145;
gint PeopleWarningLimit = 30;
gint NobodyWarningLimit = 15;
gint SuspiciousItemWarningLimit = 15;

extern "C" void
readConfig(){ 
  char name[SIZE];
  char value[SIZE];

  memset(ipaddress,0,SIZE);
  
  FILE *fp = fopen(CONFIG_PATH, "r");
  if (fp == NULL){
    return;
  }else{
    while(!feof(fp)){
      memset(name,0,SIZE);
      memset(value,0,SIZE);

      /*Read Data*/
      fscanf(fp,"%s = %s\n", name, value);
      
      if(!strcmp(name, "port_lock")){
        port_lock = atoi(value);
      }
      else if (!strcmp(name, "portnumber")){
        portnumber = atoi(value);
      }
      else if(!strcmp(name, "ipaddress")){
        strcpy(ipaddress, value);
      }
      else if(!strcmp(name, "pose_estimation_muxer_output_width")){
        pose_estimation_muxer_output_width = atoi(value);
      }
      else if(!strcmp(name, "pose_estimation_muxer_output_height")){
        pose_estimation_muxer_output_height = atoi(value);
      }
      else if(!strcmp(name, "open_send_socket_count_limit")){
        open_send_socket_count_limit = atoi(value);
      }
      else if(!strcmp(name, "send_socket_count_limit")){
        send_socket_count_limit = atoi(value);
      }
      else if(!strcmp(name, "open_hand_lock")){
        open_hand_lock = atoi(value);
      }
      else if(!strcmp(name, "PoseWarningLimit")){
        PoseWarningLimit = atoi(value);
      }
      else if(!strcmp(name, "LeftArmMin")){
        LeftArmMin = atoi(value);
      }
      else if(!strcmp(name, "LeftArmMax")){
        LeftArmMax = atoi(value);
      }
      else if(!strcmp(name, "RightArmMin")){
        RightArmMin = atoi(value);
      }
      else if(!strcmp(name, "RightArmMax")){
        RightArmMax = atoi(value);
      }
      else if(!strcmp(name, "PeopleWarningLimit")){
        PeopleWarningLimit = atoi(value);
      }
      else if(!strcmp(name, "NobodyWarningLimit")){
        NobodyWarningLimit = atoi(value);
      }
      else if(!strcmp(name, "SuspiciousItemWarningLimit")){
        SuspiciousItemWarningLimit = atoi(value);
      }
    }
  }
  fclose(fp);
 
  return;
}

/*Method to parse information returned from the model*/
std::tuple<Vec2D<int>, Vec3D<float>>
parse_objects_from_tensor_meta(NvDsInferTensorMeta *tensor_meta)
{
  Vec1D<int> counts;
  Vec3D<int> peaks;

  float threshold = 0.1;
  int window_size = 5;
  int max_num_parts = 20;
  int num_integral_samples = 7;
  float link_threshold = 0.1;
  int max_num_objects = 100;

  void *cmap_data = tensor_meta->out_buf_ptrs_host[0];
  NvDsInferDims &cmap_dims = tensor_meta->output_layers_info[0].inferDims;
  void *paf_data = tensor_meta->out_buf_ptrs_host[1];
  NvDsInferDims &paf_dims = tensor_meta->output_layers_info[1].inferDims;

  /* Finding peaks within a given window */
  find_peaks(counts, peaks, cmap_data, cmap_dims, threshold, window_size, max_num_parts);
  /* Non-Maximum Suppression */
  Vec3D<float> refined_peaks = refine_peaks(counts, peaks, cmap_data, cmap_dims, window_size);
  /* Create a Bipartite graph to assign detected body-parts to a unique person in the frame */
  Vec3D<float> score_graph = paf_score_graph(paf_data, paf_dims, topology, counts, refined_peaks, num_integral_samples);
  /* Assign weights to all edges in the bipartite graph generated */
  Vec3D<int> connections = assignment(score_graph, topology, counts, link_threshold, max_num_parts);
  /* Connecting all the Body Parts and Forming a Human Skeleton */
  Vec2D<int> objects = connect_parts(connections, topology, counts, max_num_objects);
  return {objects, refined_peaks};
}

float convert_radian_to_degrees(float radian){ 
    float pi = 3.14159; 
    return (radian * (180/pi)); 
}

float get_angle(float a0, float a1, float b0, float b1)
{
    float del_y = a1-b1;
    float del_x = b0-a0;
    if (del_x == 0)
      del_x = 0.1;
    float angle = 0;
    if (del_x > 0 && del_y > 0)
      angle = convert_radian_to_degrees(atan(del_y / del_x));
    if (del_x < 0 && del_y > 0)
      angle = convert_radian_to_degrees(atan(del_y / del_x)) + 180;

    return angle;
}

extern "C" void
send_lock_socket(char buf[], bool detection){
  int sockfd = 0;
  int so_broadcast = 1;
  struct sockaddr_in server_addr, client_addr;

  /*Create an IPv4 UDP socket*/
  if((sockfd = socket(AF_INET, SOCK_DGRAM, 0))<0){
    perror("socket");
    return;
  }

  /*SO_BROADCAST: broadcast attribute*/
  if(setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &so_broadcast, sizeof(so_broadcast))<0){
    perror("setsockopt");
    return;
  }

  server_addr.sin_family = AF_INET; /*IPv4*/
  server_addr.sin_port = htons(INADDR_ANY); /*All the port*/
  server_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST); /*Broadcast address*/

  if((bind(sockfd, (struct sockaddr*)&server_addr, sizeof(struct sockaddr))) != 0){
    perror("bind");
    return;
  }

  client_addr.sin_family = AF_INET; /*IPv4*/
  client_addr.sin_port = htons(portnumber);  /*Set port number*/
  if (port_lock == 1)
  {
    client_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST); /*Set the broadcast address*/
  }
  else{
    client_addr.sin_addr.s_addr = inet_addr(ipaddress); /*Set the broadcast address*/
  }
  int clientlen = sizeof(client_addr);

  /*Use sendto() to send messages to client*/
  /*sendto() doesn't need to be connected*/
  if((sendto(sockfd, buf, strlen(buf), 0, (struct sockaddr*)&client_addr, (socklen_t)clientlen)) < 0){
    perror("sendto");
    return;
  }
  else{
    printf("send msg %s\n", buf);
    if(detection){
      PoseWarning = 0;
      PeopleWarning = 0;
      NobodyWarning = 0;
      SuspiciousItemWarning = 0;
      ShutdownCommand += 1;
      if(open_send_socket_count_limit == 1)
      {
        if(ShutdownCommand == send_socket_count_limit)
        {
          // system("killall deepstream-app");
          system("shutdown -h now");
        }
      }
    }
  }

  close(sockfd);  /*close socket*/
  return;
}

/* MetaData to handle drawing onto the on-screen-display */
static void
create_display_meta(Vec2D<int> &objects, Vec3D<float> &normalized_peaks, NvDsFrameMeta *frame_meta, int frame_width, int frame_height)
{
  int K = topology.size();
  int countPeople = objects.size();
  NvDsBatchMeta *bmeta = frame_meta->base_meta.batch_meta;
  NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(bmeta);
  nvds_add_display_meta_to_frame(frame_meta, dmeta);

  bool IsWarning = false;
  for (auto &object : objects)
  {
    int C = object.size();
    for (int j = 0; j < C; j++)
    {
      int k = object[j];
      if (k >= 0)
      {
        auto &peak = normalized_peaks[j][k];
        int x = peak[1] * pose_estimation_muxer_output_width;
        int y = peak[0] * pose_estimation_muxer_output_height;
        if (dmeta->num_circles == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_CircleParams &cparams = dmeta->circle_params[dmeta->num_circles];
        cparams.xc = x;
        cparams.yc = y;
        cparams.radius = 6;
        cparams.circle_color = NvOSD_ColorParams{244, 67, 54, 1};
        cparams.has_bg_color = 1;
        cparams.bg_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_circles++;
      }
    }
    for (int k = 0; k < K; k++)
    {
      int c_a = topology[k][2];
      int c_b = topology[k][3];
      if (object[c_a] >= 0 && object[c_b] >= 0)
      {
        auto &peak0 = normalized_peaks[c_a][object[c_a]];
        auto &peak1 = normalized_peaks[c_b][object[c_b]];
        if (k == 7)
        {
          float angle = get_angle(peak0[1], peak0[0], peak1[1], peak1[0]);
          // printf("left arm angle %f\n", angle);
          if (LeftArmMin < angle and angle < LeftArmMax)
            IsWarning = true;
        }
        if (k ==8)
        {
          float angle = get_angle(peak0[1], peak0[0], peak1[1], peak1[0]);
          // printf("right arm angle %f\n", angle);
          if (RightArmMin < angle and angle < RightArmMax)
            IsWarning = true;
        }
        int x0 = peak0[1] * pose_estimation_muxer_output_width;
        int y0 = peak0[0] * pose_estimation_muxer_output_height;
        int x1 = peak1[1] * pose_estimation_muxer_output_width;
        int y1 = peak1[0] * pose_estimation_muxer_output_height;
        if (dmeta->num_lines == MAX_ELEMENTS_IN_DISPLAY_META)
        {
          dmeta = nvds_acquire_display_meta_from_pool(bmeta);
          nvds_add_display_meta_to_frame(frame_meta, dmeta);
        }
        NvOSD_LineParams &lparams = dmeta->line_params[dmeta->num_lines];
        lparams.x1 = x0;
        lparams.x2 = x1;
        lparams.y1 = y0;
        lparams.y2 = y1;
        lparams.line_width = 3;
        //g_print("%d\n",k);
        lparams.line_color = NvOSD_ColorParams{0, 255, 0, 1};
        dmeta->num_lines++;
      }
      else if(open_hand_lock == 1)
      {
        if((k == 7 || k == 8) && (object[c_a] < 0 && object[c_b] < 0))
        {
          IsWarning = true;
        }
      }
    }
  }

  if(IsWarning)
  {
    PoseWarning++;
    // printf("Pose Warning : %d \n", PoseWarning);
  }
  else
  {
    PoseWarning=0;
  }

  if(countPeople > 1)
  {
    NobodyWarning=0;
    PeopleWarning++;
    // printf("Over People Warning : %d \n", PeopleWarning);
  }
  else if(countPeople == 0)
  {
    PeopleWarning=0;
    NobodyWarning++;
    // printf("Nobody Warning : %d \n", NobodyWarning);
  }
  else
  {
    PeopleWarning=0;
    NobodyWarning=0;
  }

  if(PoseWarning == PoseWarningLimit || PeopleWarning == PeopleWarningLimit || NobodyWarning == NobodyWarningLimit)
  {
    send_lock_socket(lockbuf , true);
  }
}

extern "C" void
pose_meta_data(NvDsBatchMeta *batch_meta)
{
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_user = NULL;
   
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        for (l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
            {
                NvDsInferTensorMeta *tensor_meta =
                    (NvDsInferTensorMeta *)user_meta->user_meta_data;
                Vec2D<int> objects;
                Vec3D<float> normalized_peaks;
                tie(objects, normalized_peaks) = parse_objects_from_tensor_meta(tensor_meta);
                create_display_meta(objects, normalized_peaks, frame_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);
            }
        }
    }
    return;
}

extern "C" void
object_meta_data0(NvDsBatchMeta *batch_meta)
{
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
   
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        bool IsWarning = false;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)l_obj->data;
            if (obj_meta->obj_label[0] != '\0')
            {
              if(!strcmp(obj_meta->obj_label,"SuspiciousItem"))
              {
                IsWarning = true;
              }
            }
        }
        if(IsWarning)
        {
          SuspiciousItemWarning++;
          // printf("Suspicious Item Warning : %d \n", SuspiciousItemWarning);
        }
        else
        {
          SuspiciousItemWarning = 0;
        }
    }

    if(SuspiciousItemWarning == SuspiciousItemWarningLimit)
    {
      send_lock_socket(lockbuf , true);
    }
    return;
}