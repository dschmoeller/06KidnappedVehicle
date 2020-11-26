/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <list>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::cout; 
using std::endl; 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Number of particles 
  num_particles = 60;  

  // Define Gaussian distribution as particle generator
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  // Create particles and initialize them with GPS information
  for (int i = 0; i < num_particles; i++){
    // Create and init particle
    Particle p = Particle(); 
    p.id = i; 
    p.x = dist_x(gen); 
    p.y = dist_y(gen); 
    p.theta = dist_theta(gen); 
    p.weight = 1.0;
    // Add particle and weight to their respective sets
    particles.push_back(p); 
    weights.push_back(p.weight); 
  }
  // Set initialization flag
  is_initialized = true;  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
   // For each particle, calculate Particle Movement (Based on velocity and yaw rate measurements)
   std::mt19937 gen; 
   for (auto& p : particles){
     // Take care of division by zero
     double x_f, y_f, theta_f;  
     // Use linear model if yaw rate is close to zero
     if (yaw_rate >= 0 && yaw_rate < 0.001){
       x_f = p.x + velocity*delta_t*cos(p.theta); 
       y_f = p.y + velocity*delta_t*sin(p.theta); 
       theta_f = p.theta; 
     }
     // Use general model if yaw rate is greater than zero
     else {
       x_f = p.x + (velocity/yaw_rate)*(sin(p.theta + yaw_rate*delta_t) - sin(p.theta)); 
       y_f = p.y + (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta + yaw_rate*delta_t)); 
       theta_f = p.theta + yaw_rate*delta_t;
     }
     
     // Add Gaussian noise and update particle
     std::normal_distribution<double> dist_x(x_f, std_pos[0]);
     std::normal_distribution<double> dist_y(y_f, std_pos[1]);
     std::normal_distribution<double> dist_theta(theta_f, std_pos[2]);
     p.x = dist_x(gen); 
     p.y = dist_y(gen); 
     p.theta = dist_theta(gen);   
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  // Check each expected LM for the corresponding observation
  // If there's more observations than LMs, observations are gonna be unassigned
  // Later for the probability calculation, this means that the prob of this 
  // measurement got to be 0
  for (auto pred_LM : predicted){
    // Calculate error (i.e. distance) against each observation
    vector < std::tuple<double, LandmarkObs*> > dist_list; 
    for (auto& obs : observations){
      // Check if observation has not been assigned to a LM already
      if (obs.id == 0) {
        double eucl_dist = dist(pred_LM.x, pred_LM.y, obs.x, obs.y);
        dist_list.push_back(std::make_tuple(eucl_dist, &obs));    
      }     
    }
    // Sort the distance/error list in ascending order
    // Lowest error element on the top of the vector
    // Assign current LM id to lowest error observation
    if (dist_list.size() != 0){
      sort(dist_list.begin(), dist_list.end()); 
      std::get<1>(dist_list[0])->id = pred_LM.id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  // Calculate observation probabilities for each particle
  for (auto& p : particles){
    // (1) TRANSFORM OBSERVATION FROM PARTICLE COS INTO WORLD COS
    vector<LandmarkObs> obs_trans; 
    for (auto obs : observations){
      // Apply transormation matrix
      double x_m = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y; 
      double y_m = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y; 
      // Store transformed coordinates in obs_trans
      LandmarkObs LM_obs_global = LandmarkObs(); 
      LM_obs_global.x = x_m; 
      LM_obs_global.y = y_m; 
      LM_obs_global.id = 0; 
      obs_trans.push_back(LM_obs_global); 
    }
    // (2) ESTIMATE WHICH LANDMARKS THE PARTICLE IS SUPPOSED TO OBSERVE
    // Define particle sensor range
    double x_min = p.x - sensor_range; 
    double x_max = p.x + sensor_range; 
    double y_min = p.y - sensor_range; 
    double y_max = p.y + sensor_range; 
    // Cut the LMs of interest from the map (Check whether they are within particle range)
    // and push them to the LMs_in_range vector
    vector<LandmarkObs> LMs_in_range; 
    for (auto LM : map_landmarks.landmark_list){
      if (LM.x_f > x_min && LM.x_f < x_max && LM.y_f > y_min && LM.y_f < y_max){
        LandmarkObs LM_in_range = LandmarkObs();
        LM_in_range.x = LM.x_f; 
        LM_in_range.y = LM.y_f; 
        LM_in_range.id = LM.id_i;  
        LMs_in_range.push_back(LM_in_range); 
      }
    }
    // (3) ASSOCIATE OBSERVATIONS WITH LANDMARKS 
    dataAssociation(LMs_in_range, obs_trans); 
    // (4) CALCULATE PROBABILITIES
    //  Calculate single landmark observation probability using mulitivariate gaussian
    vector <double> LM_obs_probs; 
    for (auto obs : obs_trans){
      int assigned_LM_id = obs.id;
      double LM_obs_prob = 1;  
      // Could this observation be matched with a landmark
      // If yes, calucalte probaility
      // If not, probability is zero
      if(assigned_LM_id != 0){
        double mu_x = map_landmarks.landmark_list[assigned_LM_id - 1].x_f; 
        double mu_y = map_landmarks.landmark_list[assigned_LM_id - 1].y_f;
        LM_obs_prob = multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, mu_x, mu_y);
        // Deal with numerical boundaries
        if (LM_obs_prob == 0){
          LM_obs_prob = 1e-10; 
        }     
      }
      else{
        LM_obs_prob = 0; 
      }
      // Push single prob onto vector
      LM_obs_probs.push_back(LM_obs_prob); 
    }
    // Calculate entire probability 
    double weight = 1; 
    for (auto prob : LM_obs_probs){
      weight*=prob; 
    }
    // Update particle weight
    p.weight = weight; 
  }
}

void ParticleFilter::resample() {
  // Fetch particle probabilities(weights) and normalize them 
  // so they scale up to [0, 1000] 
  int cnt = 0;
  double sum_of_weights = 0; 
  for (auto& p : particles){
    weights[cnt] = p.weight;
    sum_of_weights += p.weight;  
    cnt++; 
  }
  double scale_factor = 1000.0 / sum_of_weights; 
  std::list<int> scaled_weights;
  for (auto w : weights){
    double scaled_weight = scale_factor*w; 
    scaled_weights.push_back(int(scaled_weight));  
  }
  // Use discrete distrubtion  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(scaled_weights.begin(), scaled_weights.end());
  std::map<int, int> m;
  vector<Particle> particles_updated; 
  for(int n=0; n<num_particles; ++n) {
      ++m[d(gen)];   
  }
  int particle_index_cnt = 0; 
  for(auto p : m) { 
    int particle_id = p.first; 
    int num_iterations = p.second; 
    for (int i = 0; i < num_iterations; i++){
      Particle p = Particle();
      p.id = particle_index_cnt; 
      p.x = particles[particle_id].x; 
      p.y = particles[particle_id].y; 
      p.theta = particles[particle_id].theta;
      p.weight = particles[particle_id].weight;
      particles_updated.push_back(p);  
      particle_index_cnt++;  
    } 
  }
  // Use resampled particles as current particle set
  particles = particles_updated;
  // Update particle weights 
  int weight_cnt = 0; 
  for (auto p : particles){
    weights[weight_cnt] = p.weight; 
    weight_cnt++; 
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}