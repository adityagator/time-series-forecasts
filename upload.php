<!DOCTYPE html>
<html lang="en">
    <head>
    <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Time Series Forecast</title>
        <link rel="stylesheet" href="https://unpkg.com/aos@next/dist/aos.css" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Nanum+Gothic">
        <link rel="stylesheet" href="main.css">
        <script src="main.js"></script>
    </head>

    <body>
    <form action="upload.php" method="post" enctype="multipart/form-data">
                <label for="input_file">Acceptable file types include .csv and .xlsx</label><br>
                <input type="file" id="fileToUpload" name="fileToUpload"><br><br>
                <input type="submit" value="Upload File" name="submit">
    </form>

    <?php
$target_dir = "input/";
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image
 if(isset($_POST["submit"])) {
   // Check if file already exists
   if (file_exists($target_file)) {
       echo "Sorry, file already exists.";
       $uploadOk = 0;
   }
   // Check file size
   if ($_FILES["fileToUpload"]["size"] > 500000) {
       echo "Sorry, your file is too large.";
       $uploadOk = 0;
   }
   // Allow csv and xlsx file formats only
   if($imageFileType != "csv" && $imageFileType != "xlsx") {
       echo "Sorry, only csv & xlsx files are allowed.";
       $uploadOk = 0;
   }
   // Check if $uploadOk is set to 0 by an error
   if ($uploadOk == 0) {
       echo "Sorry, your file was not uploaded.";
   } else {
       if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {
           echo "The file ". basename( $_FILES["fileToUpload"]["name"]). " has been uploaded.";
       } else {
           echo "Sorry, there was an error uploading your file.";
       }
   }
 }

?>
    </body>
    </html>
