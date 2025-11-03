import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:mobile_app/App_Color.dart';
import 'package:mobile_app/components/button.dart';
import 'package:mobile_app/components/input_text_field.dart';

import '../../../P.dart';

class ChangePassPage extends StatelessWidget {
  ChangePassPage({super.key});

  final TextEditingController _passController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: AppColors.green,
        title: Text(
          "Confirm Password",
          style: TextStyle(color: AppColors.white, fontSize: 20),
        ),
        centerTitle: true,
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(height: 20),
          InputTextField(
            controller: _passController,
            textWarning: "Enter your password",
            hintText: "Password",
            obs: true,
            readOnly: false,
          ),
          ButtonAuth(
            content: "Confirm",
            onTap: () {
              P.auth.authenticationPassword(
                FirebaseAuth.instance.currentUser!.email!,
                _passController.text,
              );
            },
          ),
        ],
      ),
    );
  }
}
