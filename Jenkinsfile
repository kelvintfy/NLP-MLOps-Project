pipeline {
    agent { docker { image 'python:3.7.2' } }
    stages {
        stage('Build') {
            steps {
                echo 'Installing Python dependencies . . .'
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
                echo 'Installations completed !'
            }
        }
        stage('Test') {
            steps {
                echo 'Running unit tests . . .'
                sh 'python test.py'
                echo 'NO error is found.'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Ready to deploy!'
            }
        }
    }
}